"""End-to-end graph tests: unfused reference vs fused execution.

The unfused graph is evaluated with the generic per-node interpreter while
the fused graph runs the standalone attention operator; the two execution
paths share no VJP code beyond NumPy primitives.  Every output value and the
joint VJP over all floating inputs are compared under the contract tolerances.
"""
import numpy as np
import pytest

import graph
from examples import (model, inputs, single_block_model, single_block_feeds,
                      two_block_model, two_block_feeds,
                      random_graph, random_feeds)
from optimizer import optimize

TOL64 = dict(atol=1e-8, rtol=1e-6)
TOL32 = dict(atol=2e-5, rtol=2e-4)


def run_pair(spec, feeds, tol):
    plain, tape = graph.evaluate(spec, feeds)
    fused = optimize(spec, feeds)
    got, ftape = graph.evaluate(fused, feeds)
    assert set(plain) == set(got)
    for name in plain:
        np.testing.assert_allclose(got[name], plain[name], **tol)

    seeds = {name: np.random.default_rng(abs(hash(name)) % 2**32)
             .normal(size=value.shape)
             for name, value in plain.items()}
    pg = graph.backward(spec, tape, seeds)
    fg = graph.backward(fused, ftape, seeds)
    assert set(pg) == set(fg)
    for name in pg:
        assert pg[name].shape == fg[name].shape == feeds[name].shape
        np.testing.assert_allclose(fg[name], pg[name], **tol)
    return plain, pg, fused


@pytest.mark.parametrize('dtype,tol', [(np.float64, TOL64),
                                       (np.float32, TOL32)])
@pytest.mark.parametrize('scale_q', [False, True])
@pytest.mark.parametrize('n_blocks', [1, 2, 3])
def test_random_graphs(dtype, tol, scale_q, n_blocks):
    rng = np.random.default_rng(hash((dtype, scale_q, n_blocks)) % 2**32)
    for trial in range(12):
        spec = random_graph(rng, n_blocks=n_blocks,
                            scale_q_prob=1.0 if scale_q else 0.0,
                            bias_prob=0.9, prob_output_prob=0.7)
        assert len(spec['nodes']) <= 60
        feeds = random_feeds(rng, dtype)
        plain, pg, fused = run_pair(spec, feeds, tol)
        assert sum(n['op'] == 'attention' for n in fused['nodes']) == n_blocks
        # Side-output probability rows survive with the zero-row rule.
        for name in plain:
            if name.endswith('prob'):
                sums = plain[name].sum(axis=-1)
                live = np.broadcast_to(
                    feeds['allow'].any(axis=-1), sums.shape)
                np.testing.assert_allclose(sums[live], 1.0, **tol)
                assert np.all(sums[~live] == 0.0)


def test_all_false_mask_graph():
    spec = single_block_model()
    feeds = single_block_feeds(np.float64)
    feeds['allow'] = np.zeros_like(feeds['allow'])
    plain, pg, fused = run_pair(spec, feeds, TOL64)
    assert np.all(plain['out'] == 0.0) and np.all(plain['prob'] == 0.0)
    for name, value in pg.items():
        assert np.all(value == 0.0), name


def test_no_bias_graph_fuses_and_matches():
    rng = np.random.default_rng(99)
    spec = random_graph(rng, n_blocks=1, bias_prob=0.0)
    feeds = random_feeds(rng, np.float64)   # bias feed exists but is unused
    plain, _, fused = run_pair(spec, feeds, TOL64)
    attn = [n for n in fused['nodes'] if n['op'] == 'attention']
    assert len(attn) == 1 and len(attn[0]['inputs']) == 4


def test_legacy_model_all_flags():
    for shared in (False, True):
        for bias in (False, True):
            for dynamic in (False, True):
                for alternate in (False, True):
                    spec = model(shared=shared, bias=bias,
                                 dynamic=dynamic, alternate=alternate)
                    feeds = inputs(all_masked=True,
                                   scale=0.4 + 0.1 * dynamic)
                    if not bias:
                        feeds = {k: v for k, v in feeds.items() if k != 'bias'}
                    plain, pg, fused = run_pair(spec, feeds, TOL64)
                    # fixed_scale hint must never replace the runtime input.
                    assert all('fixed_scale' not in n
                               for n in fused['nodes'])
                    assert 'scale' in pg and np.any(pg['scale'] != 0.0)


def test_two_blocks_shared_parameters_accumulate():
    spec, feeds = two_block_model(), two_block_feeds(np.float64)
    plain, pg, fused = run_pair(spec, feeds, TOL64)
    assert sum(n['op'] == 'attention' for n in fused['nodes']) == 2
    # The shared scale/bias/q/k/v grads must be non-zero: both blocks feed
    # the scalar loss, so accumulation actually happened.
    for name in ['scale', 'bias', 'q', 'k', 'v']:
        assert np.any(pg[name] != 0.0), name
    # Gradient of a sum-of-everything scalar loss is ones at the output.
    seeds = {'loss': np.ones(())}
    _, tape = graph.evaluate(spec, feeds)
    g = graph.backward(spec, tape, seeds)
    assert set(g) == {'q', 'k', 'v', 'scale', 'bias'}


def test_probability_side_output_feeds_second_loss():
    """The attention probabilities must remain usable as another loss input."""
    spec = single_block_model(prob_output=True)
    feeds = single_block_feeds(np.float64)
    nodes = spec['nodes'] + [
        {'id': 'p_loss', 'op': 'sum', 'inputs': ['prob']},
        {'id': 'o_loss', 'op': 'sum', 'inputs': ['out']},
        {'id': 'total', 'op': 'add', 'inputs': ['p_loss', 'o_loss']},
    ]
    spec2 = {'nodes': nodes, 'outputs': ['total', 'prob']}
    plain, pg, fused = run_pair(spec2, feeds, TOL64)
    # The prob chain stays alive because prob is an output and a loss input.
    assert any(n['id'] == 'prob' for n in fused['nodes'])
    assert any(n['op'] == 'attention' for n in fused['nodes'])


def test_source_spec_and_feeds_are_immutable():
    import copy
    spec = single_block_model()
    feeds = single_block_feeds(np.float64)
    snapshot_nodes = copy.deepcopy(spec['nodes'])
    snapshot_feeds = {k: v.copy() for k, v in feeds.items()}
    fused = optimize(spec, feeds)
    assert spec['nodes'] == snapshot_nodes
    graph.evaluate(spec, feeds)
    for k in feeds:
        np.testing.assert_array_equal(feeds[k], snapshot_feeds[k])
    assert fused is not spec


def test_repeated_optimize_is_idempotent():
    spec = two_block_model()
    feeds = two_block_feeds(np.float64)
    once = optimize(spec, feeds)
    twice = optimize(once, feeds)
    thrice = optimize(twice, feeds)
    assert [n['id'] for n in twice['nodes']] == [n['id'] for n in once['nodes']]
    assert thrice == twice
    # No dangling references and the executed graph stays correct.
    run_pair(spec, feeds, TOL64)


def test_reuse_graph_with_new_shapes_and_values():
    spec = single_block_model()
    opt = optimize(spec)                       # no feeds available yet
    for seed in range(5):
        feeds = single_block_feeds(np.float64, seed=seed,
                                   scale=0.1 + 0.05 * seed)
        # Coherent re-slicing: B, L, S all change between runs.
        b = 1 + seed % 2
        l = 2 + seed % 2
        s = 3 + seed % 3
        feeds['q'] = feeds['q'][:b, :l]
        feeds['k'] = feeds['k'][:b, :s]
        feeds['v'] = feeds['v'][:b, :s]
        feeds['bias'] = feeds['bias'][:, :l, :s]
        feeds['allow'] = feeds['allow'][:b, :l, :s]
        # The pre-optimized graph and a fresh optimize() both stay correct.
        run_pair(spec, feeds, TOL64)
        run_pair(opt, feeds, TOL64)


def test_non_fusible_graph_is_preserved():
    # Broadcast (non-scalar) mul structurally inside the chain must stay
    # as ordinary graph ops rather than be mis-fused.
    rng = np.random.default_rng(55)
    feeds = random_feeds(rng, np.float64)
    b, l, s = feeds['q'].shape
    feeds['gate'] = rng.normal(size=(1, l, s))
    spec = {
        'nodes': [
            {'id': name, 'op': 'input'}
            for name in ['q', 'k', 'v', 'scale', 'bias', 'allow', 'gate']
        ] + [
            {'id': 'kt', 'op': 'transpose', 'inputs': ['k']},
            {'id': 'dot', 'op': 'matmul', 'inputs': ['q', 'kt']},
            {'id': 'gated', 'op': 'mul', 'inputs': ['dot', 'gate']},
            {'id': 'biased', 'op': 'add', 'inputs': ['gated', 'bias']},
            {'id': 'masked', 'op': 'where_mask',
             'inputs': ['biased', 'allow']},
            {'id': 'prob', 'op': 'softmax_last', 'inputs': ['masked']},
            {'id': 'out', 'op': 'matmul', 'inputs': ['prob', 'v']},
        ],
        'outputs': ['out']}
    fused = optimize(spec, feeds)
    assert all(n['op'] != 'attention' for n in fused['nodes'])
    # Semantics preserved even though nothing fused.
    run_pair(spec, feeds, TOL64)


def test_distinct_subgraphs_keep_distinct_caches():
    """Two blocks with different masks/bias must not share a fused cache."""
    spec = two_block_model()
    feeds = two_block_feeds(np.float64)
    fused = optimize(spec, feeds)
    _, caches = graph.evaluate(fused, feeds)[1]
    attn_ids = [n['id'] for n in fused['nodes'] if n['op'] == 'attention']
    assert len(attn_ids) == 2
    p0, p1 = (caches[i].p for i in attn_ids)
    assert p0 is not p1 and not np.array_equal(p0, p1)


def test_graph_finite_difference_scalar_loss():
    """FD check through the whole fused graph (different check than pair test)."""
    spec = single_block_model()
    feeds = single_block_feeds(np.float64)
    fused = optimize(spec, feeds)
    _, tape = graph.evaluate(fused, feeds)

    def loss(mut_feeds):
        out, _ = graph.evaluate(fused, mut_feeds)
        return float(np.sum(out['out']))

    seeds = {'out': np.ones((feeds['q'].shape[0], feeds['q'].shape[1],
                             feeds['v'].shape[2]))}
    analytic = graph.backward(fused, tape, seeds)
    h = 1e-6
    for name in ['q', 'k', 'v', 'scale', 'bias']:
        x = feeds[name]
        fd = np.zeros_like(x, dtype=np.float64)
        for i in range(fd.size):
            flat = x.reshape(-1)
            old = flat[i]
            flat[i] = old + h
            plus = loss(feeds)
            flat[i] = old - h
            minus = loss(feeds)
            fd.reshape(-1)[i] = (plus - minus) / (2 * h)
            flat[i] = old
        np.testing.assert_allclose(analytic[name], fd, atol=1e-7, rtol=1e-6)


# ---------------------------------------------------------------------------
# Malformed graphs and feeds
# ---------------------------------------------------------------------------

def test_deterministic_shape_coverage():
    """Sweep the full legal size range from the contract (B<=3, L,S<=8,
    D,E<=6) with deterministic shapes, exercising both fusion patterns,
    every broadcast style and fully-false masks."""
    idx = 0
    for b in (1, 3):
        for l, s in ((1, 1), (8, 8), (2, 7)):
            for d, e in ((1, 1), (6, 6), (3, 2)):
                idx += 1
                rng = np.random.default_rng(1000 + idx)
                for dtype, tol in ((np.float64, TOL64), (np.float32, TOL32)):
                    q = rng.normal(size=(b, l, d)).astype(dtype)
                    k = rng.normal(size=(b, s, d)).astype(dtype)
                    v = rng.normal(size=(b, s, e)).astype(dtype)
                    scale = np.asarray(0.3, dtype=dtype)
                    style = idx % 3
                    bias = rng.normal(
                        size=((b, l, s), (1, l, s), (l, 1))[style]).astype(dtype)
                    allow_style = idx % 2
                    allow = (rng.random(((b, l, s), (1, l, s))[allow_style])
                             > 0.5)
                    if idx % 4 == 0:
                        allow = np.zeros_like(allow)
                    feeds = dict(q=q, k=k, v=v, scale=scale,
                                 bias=bias, allow=allow)
                    for scale_q in (False, True):
                        spec = single_block_model(
                            scale_q=scale_q, prob_output=(idx % 2 == 0))
                        run_pair(spec, feeds, tol)


def test_fused_chain_fully_eliminated_without_side_outputs():
    spec = single_block_model(prob_output=False)
    feeds = single_block_feeds(np.float64)
    fused = optimize(spec, feeds)
    ops = {n['op'] for n in fused['nodes']}
    assert ops == {'input', 'attention'}
    assert len([n for n in fused['nodes'] if n['op'] == 'attention']) == 1
    run_pair(spec, feeds, TOL64)


def test_reversed_mul_operand_order_fuses():
    for scale_q in (False, True):
        spec = single_block_model(scale_q=scale_q)
        target = 'sq' if scale_q else 'scores0'
        for node in spec['nodes']:
            if node['id'] == target:
                node['inputs'] = list(reversed(node['inputs']))
        feeds = single_block_feeds()
        run_pair(spec, feeds, TOL64)
        fused = optimize(spec, feeds)
        assert any(n['op'] == 'attention' for n in fused['nodes'])


def test_errors():
    good = single_block_model()
    feeds = single_block_feeds(np.float64)

    # q[B,L,D] @ v[B,S,E]: inner dims D vs S generally disagree.
    bad = {'nodes': good['nodes'] + [
        {'id': 'x', 'op': 'matmul', 'inputs': ['q', 'v']},
        {'id': 'bad_sum', 'op': 'sum', 'inputs': ['x']}],
        'outputs': ['out', 'bad_sum']}
    with pytest.raises(graph.GraphError):
        graph.evaluate(bad, feeds)                          # shape mismatch

    bad = {'nodes': [{'id': 'a', 'op': 'matmul', 'inputs': ['q', 'k']},
                     {'id': 'q', 'op': 'input'}],
           'outputs': ['a']}
    with pytest.raises(graph.GraphError):
        graph.evaluate(bad, feeds)                          # non-topological

    bad = {'nodes': [{'id': 'q', 'op': 'input'},
                     {'id': 'q', 'op': 'input'}],
           'outputs': ['q']}
    with pytest.raises(graph.GraphError):
        graph.evaluate(bad, feeds)                          # duplicate id

    bad = {'nodes': good['nodes'], 'outputs': ['ghost']}
    with pytest.raises(graph.GraphError):
        graph.evaluate(bad, feeds)                          # unknown output

    bad = {'nodes': good['nodes'] + [
        {'id': 'dangling', 'op': 'sum', 'inputs': ['ghost']}],
           'outputs': ['out']}
    with pytest.raises(graph.GraphError):
        graph.evaluate(bad, feeds)                          # unknown reference

    bad_feeds = dict(feeds, scale=np.ones((1, 1, 1)))
    fused = optimize(good, feeds)
    with pytest.raises(graph.GraphError):
        graph.evaluate(fused, bad_feeds)                    # non-0-D scale

    bad_feeds = dict(feeds, q=feeds['q'].astype(np.float32))
    with pytest.raises(graph.GraphError):
        graph.evaluate(good, bad_feeds)                     # mixed dtypes

    bad_feeds = dict(feeds, allow=feeds['allow'].astype(np.float64))
    with pytest.raises(graph.GraphError):
        graph.evaluate(good, bad_feeds)                     # non-bool mask

    missing = {k: v for k, v in feeds.items() if k != 'q'}
    with pytest.raises(graph.GraphError):
        graph.evaluate(good, missing)                       # missing feed

    with pytest.raises(graph.GraphError):
        graph.backward(good, graph.evaluate(good, feeds)[1],
                       {'nope': np.ones(1)})                # bad seed name
