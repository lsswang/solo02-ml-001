"""Small differentiable tensor-DAG interpreter with reverse-mode autodiff.

Node protocol (nodes are dicts in topological order, ids unique):

* ``input``                 -- value comes from the feed dictionary
* ``transpose``             -- swap the last two axes
* ``matmul``                -- batched (3-D) NumPy matmul
* ``mul`` / ``add``         -- element-wise NumPy broadcast
* ``where_mask``            -- where(allow, x, -inf)
* ``softmax_last``          -- softmax over the last axis; all-masked rows
                               (all -inf) normalise to an all-zero row
* ``sum``                   -- sum every element into a 0-D array
* ``attention``             -- fused masked attention, see :mod:`attention`

``evaluate`` never mutates the spec or the feeds.  ``backward`` returns the
joint VJP over *every* floating-point input node, accumulating contributions
from all branches and all outputs.  Boolean input nodes are not differentiated.
"""
import numpy as np
import attention


class GraphError(ValueError):
    """Raised for malformed graphs or shape-incompatible feeds."""


def unbroadcast(g, shape):
    """Reduce a gradient produced under broadcasting back to ``shape``."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


def softmax(x):
    """Numerically stable last-axis softmax with zero rows for all -inf."""
    maximum = np.max(x, axis=-1, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0)
    exp = np.exp(x - safe)
    total = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, total, out=np.zeros_like(exp), where=total != 0)


# ---------------------------------------------------------------------------
# Validation / shape inference
# ---------------------------------------------------------------------------

def _matmul_shape(a, b):
    if len(a) < 1 or len(b) < 1:
        raise GraphError('matmul requires at least 1-D operands')
    if a[-1] != b[-2]:
        raise GraphError(
            f'matmul inner dimensions disagree: {a} vs {b}')
    batch = np.broadcast_shapes(a[:-2], b[:-2])
    return batch + (a[-2], b[-1])


def validate(spec, feeds=None):
    """Check structure (always) and, with feeds, shapes and dtypes.

    Returns the inferred shape dictionary when ``feeds`` is given.
    """
    nodes = spec.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise GraphError('spec must contain a non-empty node list')

    seen = set()
    for n in nodes:
        ident = n.get('id')
        if not isinstance(ident, str):
            raise GraphError('every node needs a string id')
        if ident in seen:
            raise GraphError(f'duplicate node id: {ident}')
        seen.add(ident)

    by_id = {n['id']: n for n in nodes}
    order = {}
    shapes = {}
    float_dtype = None

    def require_inputs(n, count):
        ids = n.get('inputs', [])
        if len(ids) != count:
            raise GraphError(
                f"node {n['id']} op {n['op']} expects {count} inputs, "
                f'got {len(ids)}')
        for ref in ids:
            if ref not in by_id:
                raise GraphError(f"node {n['id']} references unknown id {ref}")
        return ids

    for index, n in enumerate(nodes):
        ident, op = n['id'], n.get('op')
        ids = n.get('inputs', [])
        for ref in ids:
            if ref not in by_id:
                raise GraphError(
                    f"node {ident} references unknown id {ref}")
        for ref in ids:                       # topological ordering check
            if ref not in order:
                raise GraphError(
                    f'node {ident} is not in topological order '
                    f'(input {ref} appears later or is missing)')
        order[ident] = index

        if op == 'input':
            if feeds is None:
                continue
            if ident not in feeds:
                raise GraphError(f'missing feed for input node: {ident}')
            value = np.asarray(feeds[ident])
            shapes[ident] = value.shape
            if value.dtype.kind == 'f':
                if float_dtype is None:
                    float_dtype = value.dtype
                elif value.dtype != float_dtype:
                    raise GraphError(
                        f'all floating inputs must share one dtype; found '
                        f'{float_dtype} and {value.dtype} on {ident}')
            continue

        if op == 'transpose':
            ref = require_inputs(n, 1)[0]
            if feeds is not None:
                shape = shapes[ref]
                if len(shape) < 2:
                    raise GraphError(f'transpose node {ident} needs >= 2 axes')
                shapes[ident] = shape[:-2] + (shape[-1], shape[-2])
        elif op == 'matmul':
            refs = require_inputs(n, 2)
            if feeds is not None:
                shapes[ident] = _matmul_shape(shapes[refs[0]], shapes[refs[1]])
        elif op in ('mul', 'add'):
            refs = require_inputs(n, 2)
            if feeds is not None:
                try:
                    shapes[ident] = np.broadcast_shapes(
                        shapes[refs[0]], shapes[refs[1]])
                except ValueError as exc:
                    raise GraphError(
                        f'{op} node {ident} has incompatible shapes: {exc}')
        elif op == 'where_mask':
            refs = require_inputs(n, 2)
            if feeds is not None:
                try:
                    shapes[ident] = np.broadcast_shapes(
                        shapes[refs[0]], shapes[refs[1]])
                except ValueError as exc:
                    raise GraphError(
                        f'where_mask node {ident} has incompatible shapes: '
                        f'{exc}')
                if np.asarray(feeds[refs[1]]).dtype.kind != 'b':
                    raise GraphError(
                        f'where_mask node {ident}: mask input must be boolean')
        elif op == 'softmax_last':
            ref = require_inputs(n, 1)[0]
            if feeds is not None:
                if len(shapes[ref]) == 0:
                    raise GraphError(
                        f'softmax_last node {ident} needs at least one axis')
                shapes[ident] = shapes[ref]
        elif op == 'sum':
            ref = require_inputs(n, 1)[0]
            if feeds is not None:
                shapes[ident] = ()
        elif op == 'attention':
            refs = n.get('inputs', [])
            if len(refs) not in (4, 5):
                raise GraphError(
                    f'attention node {ident} expects inputs '
                    '[q, k, v, allow] or [q, k, v, allow, bias]')
            for ref in refs:
                if ref not in by_id:
                    raise GraphError(
                        f"node {ident} references unknown id {ref}")
            scale_ref = n.get('scale')
            if scale_ref is None or scale_ref not in by_id:
                raise GraphError(
                    f'attention node {ident} needs a valid "scale" input id')
            if feeds is not None:
                q_shape, k_shape, v_shape = (shapes[r] for r in refs[:3])
                if len(q_shape) != 3 or len(k_shape) != 3 or len(v_shape) != 3:
                    raise GraphError(
                        f'attention node {ident}: q, k, v must be 3-D')
                b, l, d = q_shape
                if k_shape != (b, k_shape[1], d):
                    raise GraphError(
                        f'attention node {ident}: k shape {k_shape} '
                        f'incompatible with q {q_shape}')
                if v_shape[0] != b or v_shape[1] != k_shape[1]:
                    raise GraphError(
                        f'attention node {ident}: v shape {v_shape} '
                        f'incompatible with k {k_shape}')
                try:
                    np.broadcast_shapes(shapes[refs[3]], (b, l, k_shape[1]))
                except ValueError as exc:
                    raise GraphError(
                        f'attention node {ident}: allow not broadcastable: '
                        f'{exc}')
                if np.asarray(feeds[refs[3]]).dtype.kind != 'b':
                    raise GraphError(
                        f'attention node {ident}: allow must be boolean')
                if len(refs) == 5:
                    try:
                        np.broadcast_shapes(shapes[refs[4]],
                                           (b, l, k_shape[1]))
                    except ValueError as exc:
                        raise GraphError(
                            f'attention node {ident}: bias not broadcastable: '
                            f'{exc}')
                if np.asarray(feeds[scale_ref]).ndim != 0:
                    raise GraphError(
                        f'attention node {ident}: scale must be 0-D')
                shapes[ident] = (b, l, v_shape[2])
        else:
            raise GraphError(f'unknown operation: {op}')

    outputs = spec.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        raise GraphError('spec must declare a non-empty outputs list')
    for name in outputs:
        if name not in by_id:
            raise GraphError(f'output references unknown node: {name}')

    return shapes if feeds is not None else None


# ---------------------------------------------------------------------------
# Forward / backward execution
# ---------------------------------------------------------------------------

def evaluate(spec, feeds):
    """Execute the graph; returns ``(outputs_dict, tape)`` without mutation."""
    validate(spec, feeds)
    # Defensive copies: execution must never write back into caller feeds.
    feeds = {key: np.array(value) for key, value in feeds.items()}
    values, caches = {}, {}
    by_id = {n['id']: n for n in spec['nodes']}

    for n in spec['nodes']:
        ident, op = n['id'], n['op']
        if op == 'input':
            values[ident] = feeds[ident]
            continue
        xs = [values[key] for key in n.get('inputs', [])]
        if op == 'transpose':
            value = np.swapaxes(xs[0], -1, -2)
        elif op == 'matmul':
            value = np.matmul(xs[0], xs[1])
        elif op == 'mul':
            value = xs[0] * xs[1]
        elif op == 'add':
            value = xs[0] + xs[1]
        elif op == 'where_mask':
            value = np.where(xs[1], xs[0], -np.inf)
        elif op == 'softmax_last':
            value = softmax(xs[0])
        elif op == 'sum':
            value = np.asarray(xs[0].sum())
        elif op == 'attention':
            kwargs = {}
            if len(xs) == 5:
                kwargs['bias'] = xs[4]
            value, caches[ident] = attention.forward(
                xs[0], xs[1], xs[2], feeds[n['scale']]
                if by_id[n['scale']]['op'] == 'input'
                else values[n['scale']],
                allow=xs[3], **kwargs)
        else:
            raise GraphError('unknown operation: ' + op)
        values[ident] = value

    return {name: values[name] for name in spec['outputs']}, (values, caches)


def backward(spec, tape, seeds):
    """Joint reverse-mode VJP for all floating-point input nodes.

    Cotangents from every entry of ``seeds`` are accumulated, including
    multiple branches that share a parameter.
    """
    values, caches = tape
    for name, seed in seeds.items():
        if name not in values:
            raise GraphError(f'seed references unknown output: {name}')
        if np.asarray(seed).shape != values[name].shape:
            raise GraphError(
                f'seed for {name} has shape {np.asarray(seed).shape}, '
                f'expected {values[name].shape}')

    grads = {key: np.array(value, copy=True) for key, value in seeds.items()}

    def add(key, g):
        g = unbroadcast(np.asarray(g), values[key].shape)
        grads[key] = grads[key] + g if key in grads else g

    for n in reversed(spec['nodes']):
        ident, op = n['id'], n['op']
        if op == 'input':
            continue
        if ident not in grads:
            # No cotangent reaches this node (e.g. an unseeded output whose
            # result feeds nothing else): nothing to propagate.
            continue
        g = grads[ident]
        ids = n.get('inputs', [])
        xs = [values[key] for key in ids]
        if op == 'transpose':
            add(ids[0], np.swapaxes(g, -1, -2))
        elif op == 'matmul':
            add(ids[0], np.matmul(g, np.swapaxes(xs[1], -1, -2)))
            add(ids[1], np.matmul(np.swapaxes(xs[0], -1, -2), g))
        elif op == 'mul':
            add(ids[0], g * xs[1])
            add(ids[1], g * xs[0])
        elif op == 'add':
            add(ids[0], g)
            add(ids[1], g)
        elif op == 'where_mask':
            add(ids[0], np.where(xs[1], g, 0))
        elif op == 'softmax_last':
            p = values[ident]
            add(ids[0], p * (g - (g * p).sum(axis=-1, keepdims=True)))
        elif op == 'sum':
            add(ids[0], np.broadcast_to(g, xs[0].shape))
        elif op == 'attention':
            ag = attention.vjp(caches[ident], g)
            add(ids[0], ag['q'])
            add(ids[1], ag['k'])
            add(ids[2], ag['v'])
            add(n['scale'], ag['scale'])
            if len(ids) == 5:
                add(ids[4], ag['bias'])
        else:
            raise GraphError('unknown backward operation: ' + op)

    return {n['id']: grads.get(n['id'], np.zeros_like(values[n['id']]))
            for n in spec['nodes']
            if n['op'] == 'input' and values[n['id']].dtype.kind == 'f'}
