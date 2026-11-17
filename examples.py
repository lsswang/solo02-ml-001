"""Example graphs and feed bundles.

* :func:`model` keeps the original signature (including the legacy
  ``fixed_scale`` exporter hint and the alternate scale-Q pattern).
* :func:`single_block_model` is the new training-time block: trainable 0-D
  scale, broadcastable bias, mask with fully-empty rows and a probability
  side output.
* :func:`two_block_model` shares one set of Q/K/V/scale/bias parameters
  between two attention blocks whose masks differ.
* :func:`random_graph` / :func:`random_feeds` generate arbitrary legal
  graphs for the correctness sweep (up to 60 nodes, three attention blocks).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Original example (backward compatible; used by the legacy demo)
# ---------------------------------------------------------------------------

def model(shared=False, bias=False, dynamic=False, alternate=False):
    nodes = [{'id': name, 'op': 'input'}
             for name in ['q', 'k', 'v', 'scale', 'allow']]
    nodes += [{'id': 'kt', 'op': 'transpose', 'inputs': ['k']},
              {'id': 'dot', 'op': 'matmul', 'inputs': ['q', 'kt']},
              {'id': 'scaled', 'op': 'mul', 'inputs': ['dot', 'scale']}]
    if not dynamic:
        nodes[-1]['fixed_scale'] = 0.5
    source = 'scaled'
    if alternate:
        nodes[6:8] = [{'id': 'sq', 'op': 'mul', 'inputs': ['q', 'scale']},
                      {'id': 'scaled', 'op': 'matmul',
                       'inputs': ['sq', 'kt']}]
    if bias:
        nodes.append({'id': 'bias', 'op': 'input'})
        nodes.append({'id': 'biased', 'op': 'add',
                      'inputs': [source, 'bias']})
        source = 'biased'
    nodes += [{'id': 'masked', 'op': 'where_mask',
               'inputs': [source, 'allow']},
              {'id': 'prob', 'op': 'softmax_last', 'inputs': ['masked']},
              {'id': 'out', 'op': 'matmul', 'inputs': ['prob', 'v']}]
    return {'nodes': nodes,
            'outputs': ['out', 'scaled', 'prob'] if shared else ['out']}


def inputs(all_masked=False, scale=0.5):
    allow = np.array([[[True, True, False], [True, False, True]]])
    if all_masked:
        allow[0, 1, :] = False
    return {'q': np.array([[[1., 0.], [0., 1.]]]),
            'k': np.array([[[1., 0.], [0., 1.], [1., 1.]]]),
            'v': np.array([[[2., -1.], [0., 3.], [4., 2.]]]),
            'scale': np.asarray(scale), 'allow': allow,
            'bias': np.array([[[0.1, 0.2, -0.3]]])}


# ---------------------------------------------------------------------------
# New training-time blocks
# ---------------------------------------------------------------------------

def single_block_model(scale_q=False, prob_output=True):
    """One attention block with trainable scale, bias and a prob side output.

    ``scale_q=True`` selects the equivalent ``(scale*Q) @ K^T`` ordering.
    """
    nodes = [{'id': name, 'op': 'input'}
             for name in ['q', 'k', 'v', 'scale', 'bias', 'allow']]
    nodes.append({'id': 'kt', 'op': 'transpose', 'inputs': ['k']})
    if scale_q:
        nodes += [{'id': 'sq', 'op': 'mul', 'inputs': ['q', 'scale']},
                  {'id': 'scores0', 'op': 'matmul',
                   'inputs': ['sq', 'kt']}]
    else:
        nodes += [{'id': 'dot', 'op': 'matmul', 'inputs': ['q', 'kt']},
                  {'id': 'scores0', 'op': 'mul',
                   'inputs': ['dot', 'scale']}]
    nodes += [{'id': 'scores', 'op': 'add',
               'inputs': ['scores0', 'bias']},
              {'id': 'masked', 'op': 'where_mask',
               'inputs': ['scores', 'allow']},
              {'id': 'prob', 'op': 'softmax_last', 'inputs': ['masked']},
              {'id': 'out', 'op': 'matmul', 'inputs': ['prob', 'v']}]
    outputs = ['out']
    if prob_output:
        outputs.append('prob')
    return {'nodes': nodes, 'outputs': outputs}


def single_block_feeds(dtype=np.float64, all_masked_row=True,
                       scale=0.37, seed=0):
    rng = np.random.default_rng(seed)
    b, l, s, d, e = 2, 3, 4, 3, 2
    q = rng.normal(size=(b, l, d)).astype(dtype)
    k = rng.normal(size=(b, s, d)).astype(dtype)
    v = rng.normal(size=(b, s, e)).astype(dtype)
    # Bias broadcasts over the batch axis.
    bias = rng.normal(size=(1, l, s)).astype(dtype)
    allow = rng.random((b, l, s)) > 0.4
    if all_masked_row:
        allow[0, 1, :] = False           # fully empty query row
        allow[1, 2, :] = False
    return {'q': q, 'k': k, 'v': v,
            'scale': np.asarray(scale, dtype=dtype),
            'bias': bias, 'allow': allow}


def two_block_model():
    """Two attention blocks sharing q/k/v/scale/bias, with different masks."""
    nodes = [{'id': name, 'op': 'input'}
             for name in ['q', 'k', 'v', 'scale', 'bias',
                          'allow1', 'allow2']]
    nodes.append({'id': 'kt', 'op': 'transpose', 'inputs': ['k']})
    blocks = []
    for idx, allow in enumerate(['allow1', 'allow2'], start=1):
        p = f'block{idx}_'
        nodes += [
            {'id': p + 'dot', 'op': 'matmul', 'inputs': ['q', 'kt']},
            {'id': p + 'scaled', 'op': 'mul',
             'inputs': [p + 'dot', 'scale']},
            {'id': p + 'biased', 'op': 'add',
             'inputs': [p + 'scaled', 'bias']},
            {'id': p + 'masked', 'op': 'where_mask',
             'inputs': [p + 'biased', allow]},
            {'id': p + 'prob', 'op': 'softmax_last',
             'inputs': [p + 'masked']},
            {'id': p + 'out', 'op': 'matmul',
             'inputs': [p + 'prob', 'v']},
        ]
        blocks.append(p + 'out')
    # A scalar loss over both blocks plus the shared probability side output.
    nodes += [{'id': 'loss1', 'op': 'sum', 'inputs': [blocks[0]]},
              {'id': 'loss2', 'op': 'sum', 'inputs': [blocks[1]]},
              {'id': 'loss', 'op': 'add',
               'inputs': ['loss1', 'loss2']}]
    return {'nodes': nodes,
            'outputs': ['loss', blocks[0], blocks[1], 'block1_prob']}


def two_block_feeds(dtype=np.float64, seed=1):
    rng = np.random.default_rng(seed)
    b, l, s, d, e = 2, 3, 5, 4, 3
    feeds = {
        'q': rng.normal(size=(b, l, d)).astype(dtype),
        'k': rng.normal(size=(b, s, d)).astype(dtype),
        'v': rng.normal(size=(b, s, e)).astype(dtype),
        'scale': np.asarray(0.25, dtype=dtype),
        'bias': rng.normal(size=(l, 1)).astype(dtype),   # two axes broadcast
    }
    feeds['allow1'] = rng.random((b, l, s)) > 0.3
    feeds['allow2'] = rng.random((1, l, s)) > 0.5       # broadcast mask
    feeds['allow1'][0, 0, :] = False
    return feeds


# ---------------------------------------------------------------------------
# Random legal graphs for the exhaustive sweep
# ---------------------------------------------------------------------------

def random_feeds(rng, dtype, broadcast_bias=True, broadcast_allow=True):
    b = rng.integers(1, 4)
    l = int(rng.integers(1, 9))
    s = int(rng.integers(1, 9))
    d = int(rng.integers(1, 7))
    e = int(rng.integers(1, 7))
    feeds = {
        'q': rng.normal(size=(b, l, d)).astype(dtype),
        'k': rng.normal(size=(b, s, d)).astype(dtype),
        'v': rng.normal(size=(b, s, e)).astype(dtype),
        'scale': np.asarray(rng.normal() * 0.5, dtype=dtype),
        'allow': rng.random((b, l, s)) > rng.uniform(0.0, 0.7),
    }
    if rng.random() < 0.2:
        feeds['allow'][:] = False            # fully masked graph
    if broadcast_allow and rng.random() < 0.5:
        feeds['allow'] = rng.random((1, l, s)) > 0.5
    if broadcast_bias:
        choice = rng.integers(0, 3)
        if choice == 0:
            feeds['bias'] = rng.normal(size=(b, l, s)).astype(dtype)
        elif choice == 1:
            feeds['bias'] = rng.normal(size=(1, l, s)).astype(dtype)
        else:
            feeds['bias'] = rng.normal(size=(l, 1)).astype(dtype)
    else:
        feeds['bias'] = rng.normal(size=(b, l, s)).astype(dtype)
    return feeds


def random_graph(rng, n_blocks=2, scale_q_prob=0.5, bias_prob=0.8,
                 prob_output_prob=0.5):
    """Build an unfused graph containing 1-3 independent attention blocks."""
    nodes = [{'id': name, 'op': 'input'}
             for name in ['q', 'k', 'v', 'scale', 'bias', 'allow']]
    outputs = []
    for idx in range(n_blocks):
        p = f'a{idx}_'
        nodes.append({'id': p + 'kt', 'op': 'transpose', 'inputs': ['k']})
        if rng.random() < scale_q_prob:
            nodes += [
                {'id': p + 'sq', 'op': 'mul',
                 'inputs': ['q', 'scale']},
                {'id': p + 'scaled', 'op': 'matmul',
                 'inputs': [p + 'sq', p + 'kt']},
            ]
        else:
            nodes += [
                {'id': p + 'dot', 'op': 'matmul',
                 'inputs': ['q', p + 'kt']},
                {'id': p + 'scaled', 'op': 'mul',
                 'inputs': [p + 'dot', 'scale']},
            ]
        source = p + 'scaled'
        if rng.random() < bias_prob:
            nodes.append({'id': p + 'biased', 'op': 'add',
                          'inputs': [source, 'bias']})
            source = p + 'biased'
        nodes += [
            {'id': p + 'masked', 'op': 'where_mask',
             'inputs': [source, 'allow']},
            {'id': p + 'prob', 'op': 'softmax_last',
             'inputs': [p + 'masked']},
            {'id': p + 'out', 'op': 'matmul',
             'inputs': [p + 'prob', 'v']},
        ]
        outputs.append(p + 'out')
        if rng.random() < prob_output_prob:
            outputs.append(p + 'prob')
    # Occasionally expose an intermediate arithmetic tensor (never a masked
    # -inf tensor, per the output contract).
    if rng.random() < 0.5:
        outputs.append('a0_scaled')
    return {'nodes': nodes, 'outputs': outputs}
