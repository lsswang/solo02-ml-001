import numpy as np


def model(shared=False, bias=False, dynamic=False, alternate=False):
    nodes = [{'id': name, 'op': 'input'} for name in ['q', 'k', 'v', 'scale', 'allow']]
    nodes += [{'id': 'kt', 'op': 'transpose', 'inputs': ['k']},
              {'id': 'dot', 'op': 'matmul', 'inputs': ['q', 'kt']},
              {'id': 'scaled', 'op': 'mul', 'inputs': ['dot', 'scale']}]
    if not dynamic: nodes[-1]['fixed_scale'] = 0.5
    source = 'scaled'
    if alternate:
        nodes[6:8] = [{'id': 'sq', 'op': 'mul', 'inputs': ['q','scale']},
                      {'id': 'scaled', 'op': 'matmul', 'inputs': ['sq', 'kt']}]
    if bias:
        nodes.append({'id': 'bias', 'op': 'input'})
        nodes.append({'id': 'biased', 'op': 'add', 'inputs': [source, 'bias']})
        source = 'biased'
    nodes += [{'id': 'masked', 'op': 'where_mask', 'inputs': [source, 'allow']},
              {'id': 'prob', 'op': 'softmax_last', 'inputs': ['masked']},
              {'id': 'out', 'op': 'matmul', 'inputs': ['prob', 'v']}]
    return {'nodes': nodes, 'outputs': ['out', 'scaled', 'prob'] if shared else ['out']}


def inputs(all_masked=False, scale=0.5):
    allow = np.array([[[True, True, False], [True, False, True]]])
    if all_masked: allow[0, 1, :] = False
    return {'q': np.array([[[1., 0.], [0., 1.]]]),
            'k': np.array([[[1., 0.], [0., 1.], [1., 1.]]]),
            'v': np.array([[[2., -1.], [0., 3.], [4., 2.]]]),
            'scale': np.asarray(scale), 'allow': allow,
            'bias': np.array([[[0.1, 0.2, -0.3]]])}
