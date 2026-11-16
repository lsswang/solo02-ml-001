"""Small differentiable tensor graph, evaluated once per topological node."""
import numpy as np
import attention


def unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


def softmax(x):
    maximum = np.max(x, axis=-1, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0)
    exp = np.exp(x - safe)
    total = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, total, out=np.zeros_like(exp), where=total != 0)


def evaluate(spec, feeds):
    values, caches = {}, {}
    for n in spec['nodes']:
        op, ident = n['op'], n['id']
        xs = [values[key] for key in n.get('inputs', [])]
        if op == 'input': value = np.asarray(feeds[ident])
        elif op == 'transpose': value = np.swapaxes(xs[0], -1, -2)
        elif op == 'matmul': value = np.matmul(*xs)
        elif op == 'mul': value = xs[0] * xs[1]
        elif op == 'add': value = xs[0] + xs[1]
        elif op == 'where_mask': value = np.where(xs[1], xs[0], -np.inf)
        elif op == 'softmax_last': value = softmax(xs[0])
        elif op == 'sum': value = np.asarray(xs[0].sum())
        elif op == 'attention':
            value, caches[ident] = attention.forward(xs[0], xs[1], xs[2], n['scale'], xs[3])
        else: raise ValueError('unknown operation: ' + op)
        values[ident] = value
    return {name: values[name] for name in spec['outputs']}, (values, caches)


def backward(spec, tape, seeds):
    values, caches = tape
    grads = {key: np.asarray(value) for key, value in seeds.items()}
    def add(key, g):
        g = unbroadcast(g, values[key].shape)
        grads[key] = grads[key] + g if key in grads else g
    for n in reversed(spec['nodes']):
        ident, op = n['id'], n['op']
        if ident not in grads or op == 'input': continue
        g = grads[ident]
        ids = n.get('inputs', [])
        xs = [values[key] for key in ids]
        if op == 'transpose': add(ids[0], np.swapaxes(g, -1, -2))
        elif op == 'matmul':
            add(ids[0], np.matmul(g, np.swapaxes(xs[1], -1, -2)))
            add(ids[1], np.matmul(np.swapaxes(xs[0], -1, -2), g))
        elif op == 'mul':
            add(ids[0], g * xs[1]); add(ids[1], g * xs[0])
        elif op == 'add': add(ids[0], g); add(ids[1], g)
        elif op == 'where_mask': add(ids[0], np.where(xs[1], g, 0))
        elif op == 'softmax_last':
            p = values[ident]
            add(ids[0], p * (g - (g * p).sum(axis=-1, keepdims=True)))
        elif op == 'sum': add(ids[0], np.broadcast_to(g, xs[0].shape))
        elif op == 'attention':
            ag = attention.vjp(caches[ident], g)
            for key, name in zip(ids[:3], ['q', 'k', 'v']): add(key, ag[name])
        else: raise ValueError('unknown backward operation: ' + op)
    return {n['id']: grads.get(n['id'], np.zeros_like(values[n['id']]))
            for n in spec['nodes'] if n['op'] == 'input' and values[n['id']].dtype.kind == 'f'}
