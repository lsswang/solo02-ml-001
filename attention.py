"""Tree-attention primitive used by the original small inference models."""
import numpy as np


def forward(q, k, v, scale, allow):
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    masked = np.where(allow, scores, -np.inf)
    with np.errstate(invalid='ignore', divide='ignore'):
        exp = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
        p = exp / exp.sum(axis=-1, keepdims=True)
    return np.matmul(p, v), (q, k, v, scale, allow, p)


def vjp(cache, dout):
    q, k, v, scale, allow, p = cache
    dv = np.matmul(np.swapaxes(p, -1, -2), dout)
    dp = np.matmul(dout, np.swapaxes(v, -1, -2))
    dz = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
    dz = np.where(allow, dz, 0)
    dq = np.matmul(dz, k) * scale
    dk = np.matmul(np.swapaxes(dz, -1, -2), q) * scale
    return {'q': dq, 'k': dk, 'v': dv}
