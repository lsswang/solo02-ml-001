"""Standalone differentiable masked attention operator.

The operator is independent of the tensor graph in :mod:`graph`: it can be
called directly with NumPy arrays.  Scores are ``scale * Q @ K.T + bias``;
probabilities are normalised only over positions where ``allow`` is True.
A query row with no allowed key gets zero probabilities and a zero output,
and contributes zero gradient to every differentiable input.

Both ``forward`` and :func:`vjp` are pure functions; the opaque cache
returned by ``forward`` holds everything ``vjp`` needs.  A small stateful
:class:`Attention` wrapper is provided as an ergonomic entry point.
"""
import numpy as np

# A row with no allowed key is represented internally by an all-minus-inf
# score row; this module never lets inf/nan escape into a public output.
NEG_INF = -np.inf


def _unbroadcast(grad, shape):
    """Sum a broadcast gradient back down to ``shape`` and reshape it."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Cache:
    """Forward-state consumed by :func:`vjp`; treated as opaque by callers."""

    __slots__ = ('q', 'k', 'v', 'scale', 'qk', 'p', 'mask',
                 'bias_shape', 'bias_value', 'dtype')

    def __init__(self, q, k, v, scale, qk, p, mask, bias_shape, dtype):
        self.q = q
        self.k = k
        self.v = v
        self.scale = scale
        self.qk = qk
        self.p = p
        self.mask = mask
        self.bias_shape = bias_shape
        self.dtype = dtype


def forward(q, k, v, scale, allow=None, bias=None):
    """Compute masked attention ``PV``.

    Parameters
    ----------
    q, k, v: np.ndarray
        Query ``[B, L, D]``, key ``[B, S, D]`` and value ``[B, S, E]``.
    scale: np.ndarray | float
        Zero-dimensional differentiable scalar.
    allow: np.ndarray | None
        Boolean array broadcastable to ``[B, L, S]``; ``True`` permits
        attention.  ``None`` means every position is allowed.
    bias: np.ndarray | None
        Floating array broadcastable to ``[B, L, S]``; defaults to zero.

    Returns
    -------
    (out, cache):
        ``out`` has shape ``[B, L, E]`` and the same float dtype as ``q``.
    """
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    if not (q.dtype.kind == 'f' and q.dtype == k.dtype == v.dtype):
        raise ValueError('q, k, v must share a floating-point dtype')
    dtype = q.dtype
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError('q, k, v must be three-dimensional [B, L, ...]')

    scale_arr = np.asarray(scale, dtype=dtype)
    if scale_arr.ndim != 0:
        raise ValueError('scale must be a zero-dimensional array')

    kt = np.swapaxes(k, -1, -2)
    qk = np.matmul(q, kt)                       # [B, L, S]
    scores = qk * scale_arr
    if bias is not None:
        bias_arr = np.asarray(bias, dtype=dtype)
        bias_shape = bias_arr.shape
        scores = scores + bias_arr
    else:
        bias_arr = None
        bias_shape = None

    if allow is None:
        mask = np.ones(qk.shape, dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(allow, dtype=bool), qk.shape)

    masked = np.where(mask, scores, NEG_INF)
    row_max = np.max(masked, axis=-1, keepdims=True)
    # Empty rows have row_max == -inf; subtract zero there instead.
    safe_max = np.where(np.isfinite(row_max), row_max, 0)
    with np.errstate(invalid='ignore'):
        exps = np.exp(masked - safe_max)        # zeros on empty rows
    totals = exps.sum(axis=-1, keepdims=True)
    p = np.divide(exps, totals,
                  out=np.zeros_like(exps), where=totals != 0)
    out = np.matmul(p, v)
    cache = Cache(q, k, v, scale_arr, qk, p,
                  np.asarray(mask), bias_shape, dtype)
    cache.bias_value = bias_arr
    return out, cache


def vjp(cache, dout):
    """Reverse-mode VJP for :func:`forward`.

    ``dout`` is any cotangent with the shape of the forward output.  Returns
    gradients for ``q``, ``k``, ``v``, ``scale`` (and ``bias`` when one was
    supplied), each with the shape of the corresponding input.  Fully masked
    rows contribute nothing.
    """
    dout = np.asarray(dout, dtype=cache.dtype)
    q, k, v = cache.q, cache.k, cache.v
    p, mask = cache.p, cache.mask

    # out = p @ v
    dv = np.matmul(np.swapaxes(p, -1, -2), dout)
    dp = np.matmul(dout, np.swapaxes(v, -1, -2))

    # Softmax VJP, restricted to allowed positions.  p is exactly zero on
    # empty rows, so the whole expression vanishes there automatically.
    row_dot = (dp * p).sum(axis=-1, keepdims=True)
    dz = p * (dp - row_dot)
    dz = np.where(mask, dz, 0)

    # scores = scale * qk (+ bias)
    dscale = np.asarray((dz * cache.qk).sum(), dtype=cache.dtype)
    dqk = dz * cache.scale
    dq = np.matmul(dqk, k)
    dk = np.matmul(np.swapaxes(dqk, -1, -2), q)

    grads = {'q': dq, 'k': dk, 'v': dv, 'scale': dscale}
    if cache.bias_shape is not None:
        grads['bias'] = _unbroadcast(dz.copy(), cache.bias_shape)
    return grads


class Attention:
    """Small stateful wrapper: ``forward`` then ``backward`` on one instance."""

    def __init__(self):
        self.cache = None

    def forward(self, q, k, v, scale, allow=None, bias=None):
        out, self.cache = forward(q, k, v, scale, allow=allow, bias=bias)
        return out

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError('forward must be called before backward')
        return vjp(self.cache, dout)
