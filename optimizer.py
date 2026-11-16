"""Original rewrite for a single-consumer attention tree with fixed scale.

`fixed_scale` is supplied by the old model loader. Old models did not expose
score/probability side outputs, train scale/bias, or use a shared attention DAG.
"""
import copy


def optimize(spec):
    result = copy.deepcopy(spec)
    by_id = {node['id']: node for node in result['nodes']}
    removed = set()
    for node in result['nodes']:
        if node['op'] != 'matmul': continue
        p = by_id[node['inputs'][0]]
        if p['op'] != 'softmax_last': continue
        masked = by_id[p['inputs'][0]]
        if masked['op'] != 'where_mask': continue
        scaled = by_id[masked['inputs'][0]]
        if scaled['op'] != 'mul' or 'fixed_scale' not in scaled: continue
        dot = by_id[scaled['inputs'][0]]
        if dot['op'] != 'matmul': continue
        trans = by_id[dot['inputs'][1]]
        if trans['op'] != 'transpose': continue
        q, k, v = dot['inputs'][0], trans['inputs'][0], node['inputs'][1]
        node.update(op='attention', inputs=[q, k, v, masked['inputs'][1]], scale=scaled['fixed_scale'])
        removed.update([p['id'], masked['id'], scaled['id'], dot['id'], trans['id']])
    result['nodes'] = [n for n in result['nodes'] if n['id'] not in removed]
    return result
