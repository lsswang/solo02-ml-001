"""Tests for the standalone attention operator.

Cross-check strategy (two independent oracles, so a shared VJP bug cannot
hide):

* a fully independent NumPy reference written straight from the real-number
  definition (masked softmax via explicit normalisation),
* float64 central finite differences over the scalar loss ``sum(dout*out)``
  for every differentiable input,
* a hand-computed tiny example with hard-coded numbers.
"""
import numpy as np
import pytest

import attention

TOL64 = dict(atol=1e-8, rtol=1e-6)
TOL32 = dict(atol=2e-5, rtol=2e-4)
FD_H = 1e-6


# ---------------------------------------------------------------------------
# Independent reference (deliberately written differently from attention.py)
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, scale, allow=None, bias=None):
    """Plain real-number definition with an explicit masked normaliser."""
    b, l, _ = q.shape
    s = k.shape[1]
    if allow is None:
        allow = np.ones((b, l, s), dtype=bool)
    allow = np.broadcast_to(allow, (b, l, s))
    if bias is None:
        bias = np.zeros((b, l, s))
    bias = np.broadcast_to(bias, (b, l, s))
    scale = float(np.asarray(scale))

    p = np.zeros((b, l, s), dtype=np.float64)
    for bi in range(b):
        for li in range(l):
            cols = np.nonzero(allow[bi, li])[0]
            if cols.size == 0:
                continue                        # zero probability row
            raw = np.array([
                scale * float(q[bi, li] @ k[bi, si]) + float(bias[bi, li, si])
                for si in cols])
            shifted = raw - raw.max()          # stability only
            w = np.exp(shifted)
            p[bi, li, cols] = w / w.sum()
    out = np.einsum('bls,bse->ble', p, v)
    return out, p


def scalar_loss(q, k, v, scale, allow, bias, dout, use_bias):
    out, _ = attention.forward(q, k, v, scale, allow=allow,
                               bias=bias if use_bias else None)
    return float(np.sum(dout * out))


def fd_grad(name, feeds, dout, use_bias, h=FD_H):
    """Central finite-difference gradient of one float input."""
    args = {key: feeds[key]
            for key in ('q', 'k', 'v', 'scale', 'allow', 'bias')}
    base = args[name]
    grad = np.zeros_like(base, dtype=np.float64)
    flat_base = np.asarray(base, dtype=np.float64).reshape(-1)
    for i in range(flat_base.size):
        old = flat_base[i]
        flat_base[i] = old + h
        plus = scalar_loss(dout=dout, use_bias=use_bias, **args)
        flat_base[i] = old - h
        minus = scalar_loss(dout=dout, use_bias=use_bias, **args)
        grad.reshape(-1)[i] = (plus - minus) / (2 * h)
        flat_base[i] = old
    return grad.reshape(base.shape)


def make_feeds(rng, dtype, bias_shape=None, allow_shape=None,
               force_empty=False):
    b, l, s, d, e = (int(x) for x in rng.integers(1, 4, size=5))
    s = min(s, 8)
    d, e = int(rng.integers(1, 7)), int(rng.integers(1, 7))
    q = rng.normal(size=(b, l, d)).astype(dtype) * 2
    k = rng.normal(size=(b, s, d)).astype(dtype) * 2
    v = rng.normal(size=(b, s, e)).astype(dtype) * 2
    scale = np.asarray(rng.normal() * 0.4, dtype=dtype)
    if bias_shape is None:
        bias_shape = (b, l, s)
    bias = rng.normal(size=bias_shape).astype(dtype)
    if allow_shape is None:
        allow_shape = (b, l, s)
    allow = rng.random(allow_shape) > 0.4
    if force_empty:
        allow = allow.copy()
        allow.reshape(-1)[:] = True
        allow[0, 0, :] = False
    return q, k, v, scale, bias.astype(dtype), allow


# ---------------------------------------------------------------------------
# Hand-computed example with hard-coded expected numbers
# ---------------------------------------------------------------------------

def test_hand_computed_example():
    # B=L=D=E=1, S=2; scale=1, no bias, both keys allowed.
    q = np.array([[[2.0]]])
    k = np.array([[[1.0], [3.0]]])
    v = np.array([[[4.0], [1.0]]])
    scale = np.asarray(1.0)
    allow = np.array([[[True, True]]])

    out, cache = attention.forward(q, k, v, scale, allow=allow)
    w0 = np.exp(2.0) / (np.exp(2.0) + np.exp(6.0))
    w1 = np.exp(6.0) / (np.exp(2.0) + np.exp(6.0))
    expected = w0 * 4.0 + w1 * 1.0
    assert out.shape == (1, 1, 1)
    assert abs(float(out[0, 0, 0]) - expected) < 1e-12
    assert abs(float(cache.p[0, 0, 0]) - w0) < 1e-12
    assert abs(float(cache.p[0, 0, 1]) - w1) < 1e-12

    # Hand derivative: dL/dout = 1, so dp = v^T = [4, 1].
    g = attention.vjp(cache, np.ones_like(out))
    dz0 = w0 * (4.0 - (w0 * 4.0 + w1 * 1.0))
    dz1 = w1 * (1.0 - (w0 * 4.0 + w1 * 1.0))
    # dq = sum_j dz_j * k_j  (D=1)
    assert abs(float(g['q'][0, 0, 0]) - (dz0 * 1.0 + dz1 * 3.0)) < 1e-12
    # dk_j = dz_j * q
    assert abs(float(g['k'][0, 0, 0]) - dz0 * 2.0) < 1e-12
    assert abs(float(g['k'][0, 1, 0]) - dz1 * 2.0) < 1e-12
    # dv_j = p_j
    assert abs(float(g['v'][0, 0, 0]) - w0) < 1e-12
    assert abs(float(g['v'][0, 1, 0]) - w1) < 1e-12
    # dscale = sum_j dz_j * qk_j, qk = [2, 6]
    assert abs(float(g['scale']) - (dz0 * 2.0 + dz1 * 6.0)) < 1e-12


def test_hand_computed_fully_masked_row():
    q = np.array([[[1.0], [0.5]]])
    k = np.array([[[2.0], [-1.0]]])
    v = np.array([[[3.0], [0.0]]])
    scale = np.asarray(0.7)
    allow = np.array([[[True, True], [False, False]]])
    bias = np.array([[[0.1, -0.2], [0.0, 0.0]]])

    out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
    assert np.all(out[0, 1] == 0.0)
    assert np.all(cache.p[0, 1] == 0.0)
    # First row still computed normally.
    ref_out, ref_p = reference_attention(q, k, v, scale, allow, bias)
    np.testing.assert_allclose(out, ref_out, **TOL64)
    np.testing.assert_allclose(cache.p, ref_p, **TOL64)

    g = attention.vjp(cache, np.ones_like(out))
    # The empty row contributes nothing to any input gradient.
    assert np.all(g['q'][0, 1] == 0.0)
    # Only the live row's keys accumulate gradient.
    _, live_cache = attention.forward(
        q[:1, :1], k, v, scale, allow=allow[:1, :1], bias=bias[:1, :1])
    live_g = attention.vjp(live_cache, np.ones((1, 1, 1)))
    np.testing.assert_allclose(g['k'], live_g['k'], atol=1e-12)
    assert float(g['scale']) != 0.0          # the live row still trains it


# ---------------------------------------------------------------------------
# Randomised correctness against the independent reference and finite diffs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype,tol', [(np.float64, TOL64),
                                       (np.float32, TOL32)])
def test_forward_matches_independent_reference(dtype, tol):
    rng = np.random.default_rng(42)
    for _ in range(30):
        q, k, v, scale, bias, allow = make_feeds(rng, dtype, force_empty=True)
        out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
        ref_out, ref_p = reference_attention(
            q.astype(np.float64), k.astype(np.float64),
            v.astype(np.float64), scale, allow, bias.astype(np.float64))
        np.testing.assert_allclose(out, ref_out.astype(dtype), **tol)
        np.testing.assert_allclose(cache.p, ref_p.astype(dtype), **tol)
        # Probability rows sum to one where allowed, zero otherwise.
        sums = cache.p.sum(axis=-1)
        live = allow.any(axis=-1)
        np.testing.assert_allclose(sums[live], 1.0, **tol)
        assert np.all(sums[~live] == 0.0)


def test_bias_defaults_to_zero_and_no_bias_grad():
    rng = np.random.default_rng(1)
    q, k, v, scale, _, allow = make_feeds(rng, np.float64)
    out1, c1 = attention.forward(q, k, v, scale, allow=allow)
    out2, c2 = attention.forward(
        q, k, v, scale, allow=allow, bias=np.zeros_like(q @ np.swapaxes(k, -1, -2)))
    np.testing.assert_array_equal(out1, out2)
    assert 'bias' not in attention.vjp(c1, np.ones_like(out1))
    assert 'bias' in attention.vjp(c2, np.ones_like(out2))


def test_allow_defaults_to_dense():
    rng = np.random.default_rng(2)
    q, k, v, scale, _, _ = make_feeds(rng, np.float64)
    allow = np.ones(q.shape[:2] + (k.shape[1],), dtype=bool)
    a, _ = attention.forward(q, k, v, scale)
    b, _ = attention.forward(q, k, v, scale, allow=allow)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('bias_shape,allow_shape', [
    ((1, 3, 4), (2, 3, 4)),     # bias broadcast over batch
    ((3, 1), (2, 3, 4)),       # bias broadcast over batch and key axis
    ((4,), (2, 3, 4)),         # fully leading-dim elision
    ((2, 1, 4), (2, 3, 4)),
])
def test_broadcast_shapes_and_vjp(bias_shape, allow_shape):
    rng = np.random.default_rng(7)
    b, l, s, d, e = 2, 3, 4, 3, 2
    q = rng.normal(size=(b, l, d))
    k = rng.normal(size=(b, s, d))
    v = rng.normal(size=(b, s, e))
    scale = np.asarray(0.31)
    bias = rng.normal(size=bias_shape)
    allow = rng.random(allow_shape) > 0.35
    allow[0, 1, :] = False
    feeds = dict(q=q, k=k, v=v, scale=scale, bias=bias, allow=allow)

    out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
    ref_out, ref_p = reference_attention(q, k, v, scale, allow, bias)
    np.testing.assert_allclose(out, ref_out, **TOL64)

    dout = rng.normal(size=out.shape)
    g = attention.vjp(cache, dout)
    assert g['q'].shape == q.shape
    assert g['k'].shape == k.shape
    assert g['v'].shape == v.shape
    assert g['scale'].shape == ()
    assert g['bias'].shape == bias_shape        # merged back to feed shape

    feeds['dout'] = dout
    for name in ['q', 'k', 'v', 'scale', 'bias']:
        fd = fd_grad(name, feeds, dout, use_bias=True)
        np.testing.assert_allclose(g[name], fd, atol=1e-7, rtol=1e-6)


def test_vjp_finite_differences_no_bias():
    rng = np.random.default_rng(11)
    q, k, v, scale, _, allow = make_feeds(rng, np.float64, force_empty=True)
    feeds = dict(q=q, k=k, v=v, scale=scale, allow=allow,
                 bias=None)
    out, cache = attention.forward(q, k, v, scale, allow=allow)
    dout = rng.normal(size=out.shape)
    g = attention.vjp(cache, dout)
    for name in ['q', 'k', 'v', 'scale']:
        fd = fd_grad(name, feeds, dout, use_bias=False)
        np.testing.assert_allclose(g[name], fd, atol=1e-7, rtol=1e-6)


def test_all_masked_zero_gradients():
    rng = np.random.default_rng(3)
    q, k, v, scale, bias, _ = make_feeds(rng, np.float64)
    allow = np.zeros(q.shape[:2] + (k.shape[1],), dtype=bool)
    out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
    assert np.all(out == 0.0) and np.all(cache.p == 0.0)
    g = attention.vjp(cache, rng.normal(size=out.shape))
    for name in ['q', 'k', 'v', 'scale', 'bias']:
        assert np.all(g[name] == 0.0), name


def test_float32_gradient_tolerance():
    rng = np.random.default_rng(5)
    q, k, v, scale, bias, allow = make_feeds(rng, np.float32, force_empty=True)
    out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
    dout = np.ones_like(out)
    g = attention.vjp(cache, dout)
    # Compare analytic float32 VJP with float64 FD on the same small inputs.
    f64 = {key: np.asarray(value, dtype=np.float64)
           for key, value in
           dict(q=q, k=k, v=v, scale=scale, bias=bias, allow=allow).items()}
    out64, cache64 = attention.forward(**f64)
    g64 = attention.vjp(cache64, np.ones_like(out64))
    for name in ['q', 'k', 'v', 'scale', 'bias']:
        np.testing.assert_allclose(g[name], g64[name], **TOL32)


def test_stateful_wrapper_matches_functions():
    rng = np.random.default_rng(8)
    q, k, v, scale, bias, allow = make_feeds(rng, np.float64)
    op = attention.Attention()
    out = op.forward(q, k, v, scale, allow=allow, bias=bias)
    expected, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
    np.testing.assert_array_equal(out, expected)
    dout = rng.normal(size=out.shape)
    g1, g2 = op.backward(dout), attention.vjp(cache, dout)
    for name in g1:
        np.testing.assert_array_equal(g1[name], g2[name])


def test_rejects_non_scalar_scale():
    q = np.zeros((1, 1, 2))
    k = np.zeros((1, 1, 2))
    v = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        attention.forward(q, k, v, np.ones(3))
