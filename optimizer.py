"""Graph optimizer that fuses attention subgraphs into ``attention`` nodes.

Two equivalent legal subgraphs are fused (bias optional):

1. ``softmax_last(where_mask(mul(matmul(Q, transpose(K)), scale) [+ bias],
   allow)) @ V``
2. the pre-scaled-query form: ``softmax_last(where_mask(matmul(mul(Q,
   scale), transpose(K)) [+ bias], allow)) @ V``

Only a *zero-dimensional* scale input is a legal fusion (checked when feeds
are supplied; otherwise enforced at execution).  Other broadcast ``mul``
nodes are ordinary graph ops and are left untouched.  The exporter-only
``fixed_scale`` hint is deliberately ignored: the runtime input node always
supplies the scale and keeps its gradient.

The source spec is never mutated.  Nodes of a matched chain are removed only
when nothing outside the fusion still uses them, so side outputs (e.g. the
attention probabilities) and shared branches keep working.  Re-running
``optimize`` is idempotent and leaves no dangling references.
"""
import copy

from graph import GraphError, validate


def _match_fusion(root, by_id, shapes=None):
    """Return a fusion record for an attention output matmul, else None.

    ``shapes`` maps node ids to runtime shapes (when feeds were supplied)
    and is used to reject structurally similar but non-scalar multipliers.
    Without feeds the matcher conservatively requires the scale operand to
    be an input node; execution re-checks the 0-D requirement.
    """
    if root['op'] != 'matmul':
        return None
    prob = by_id.get(root['inputs'][0])
    v_ref = root['inputs'][1]
    if prob is None or prob['op'] != 'softmax_last':
        return None
    masked = by_id.get(prob['inputs'][0])
    if masked is None or masked['op'] != 'where_mask':
        return None
    scores_ref, allow_ref = masked['inputs']
    scores = by_id.get(scores_ref)

    bias_ref = None
    if scores is not None and scores['op'] == 'add':
        bias_ref = scores['inputs'][1]
        scores = by_id.get(scores['inputs'][0])
    if scores is None or scores['op'] not in ('mul', 'matmul'):
        return None

    def _is_scalar(scale_ref):
        if shapes is not None:
            return len(shapes[scale_ref]) == 0
        return by_id[scale_ref]['op'] == 'input'

    # Pattern A: scale * (Q @ K^T), either mul operand order.
    if scores['op'] == 'mul':
        left, right = scores['inputs']
        left_node, right_node = by_id.get(left), by_id.get(right)
        if left_node is not None and left_node['op'] == 'matmul':
            dot, scale_ref = left, right
        elif right_node is not None and right_node['op'] == 'matmul':
            dot, scale_ref = right, left
        else:
            return None
        dot_node = by_id.get(dot)
        trans = by_id.get(dot_node['inputs'][1])
        if trans is None or trans['op'] != 'transpose':
            return None
        q_ref, k_ref = dot_node['inputs'][0], trans['inputs'][0]
        internal = {prob['id'], masked['id'], scores['id'],
                    dot_node['id'], trans['id']}
    else:
        # Pattern B: (scale * Q) @ K^T, either mul operand order.
        dot_node = scores
        trans = by_id.get(dot_node['inputs'][1])
        scaled_q = by_id.get(dot_node['inputs'][0])
        if trans is None or trans['op'] != 'transpose' \
                or scaled_q is None or scaled_q['op'] != 'mul':
            return None
        left, right = scaled_q['inputs']
        if shapes is not None:
            q_ref, scale_ref = (left, right) if len(shapes[left]) > 0 \
                else (right, left)
        elif by_id[right]['op'] == 'input':
            q_ref, scale_ref = left, right
        else:
            q_ref, scale_ref = right, left
        k_ref = trans['inputs'][0]
        internal = {prob['id'], masked['id'], scaled_q['id'],
                    dot_node['id'], trans['id']}

    if not _is_scalar(scale_ref):
        # A broadcast multiplier is an ordinary graph op, not a scale.
        return None

    add_node = by_id.get(scores_ref)
    if add_node is not None and add_node['op'] == 'add':
        internal.add(add_node['id'])

    return {
        'root': root['id'],
        'q': q_ref, 'k': k_ref, 'v': v_ref,
        'scale': scale_ref, 'allow': allow_ref, 'bias': bias_ref,
        'internal': internal,
    }


def optimize(spec, feeds=None):
    """Return a new spec with every fusible attention subgraph replaced.

    When ``feeds`` is given it is used only for validation (scale rank,
    broadcast shapes); neither it nor ``spec`` is modified.
    """
    shapes = validate(spec, feeds)
    result = copy.deepcopy(spec)
    # The exporter-only hint must never influence execution: strip it from
    # every node so it cannot shadow a runtime input or its gradient.
    for node in result['nodes']:
        node.pop('fixed_scale', None)
    by_id = {n['id']: n for n in result['nodes']}

    fusions = []
    for node in result['nodes']:
        record = _match_fusion(node, by_id, shapes)
        if record is not None:
            fusions.append(record)
    if not fusions:
        # Still a deep copy: callers may freely mutate the returned spec.
        return result

    fused_roots = {f['root'] for f in fusions}
    internals = set()
    for f in fusions:
        internals |= f['internal']

    # Rewrite the output matmul of each fusion in place, keeping its id so
    # graph outputs and other consumers keep referencing a valid node.
    for f in fusions:
        node = by_id[f['root']]
        inputs = [f['q'], f['k'], f['v'], f['allow']]
        if f['bias'] is not None:
            inputs.append(f['bias'])
        node['op'] = 'attention'
        node['inputs'] = inputs
        node['scale'] = f['scale']
        node.pop('fixed_scale', None)

    # Dead-code elimination: an internal node disappears only when every
    # remaining consumer is itself a removable internal node.  Iterate to a
    # fixed point so chains shared by two fusions resolve correctly.
    def live_consumers(node_id):
        return [n['id'] for n in result['nodes']
                if node_id in n.get('inputs', [])
                or (n['op'] == 'attention' and n.get('scale') == node_id)]

    removable = set(internals) - fused_roots
    # Graph outputs are observable and therefore always live.
    removable -= set(result['outputs'])
    changed = True
    while changed:
        changed = False
        for node_id in list(removable):
            for consumer in live_consumers(node_id):
                # A live consumer (a fused root, or any node that is not
                # itself removable) keeps this node in the graph.
                if consumer in fused_roots or consumer not in removable:
                    removable.discard(node_id)
                    changed = True
                    break

    result['nodes'] = [n for n in result['nodes'] if n['id'] not in removable]

    # Structural sanity check of the rewritten graph (no feeds: shape checks
    # happen again on the first evaluate call).
    validate(result, None)
    live_ids = {n['id'] for n in result['nodes']}
    for n in result['nodes']:
        for ref in n.get('inputs', []):
            if ref not in live_ids:
                raise GraphError(
                    f'optimizer produced dangling reference {ref}')
    return result
