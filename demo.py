"""End-to-end demonstration: unfused reference vs fused execution.

Run:  python demo.py
"""
import numpy as np

import graph
from examples import (model, inputs, single_block_model, single_block_feeds,
                      two_block_model, two_block_feeds)
from optimizer import optimize


def show(title, spec, feeds, dtype_name):
    print(f'\n=== {title} ({dtype_name}) ===')
    plain, tape = graph.evaluate(spec, feeds)
    fused = optimize(spec, feeds)
    got, ftape = graph.evaluate(fused, feeds)

    executed = [n['id'] for n in fused['nodes']]
    print('nodes actually executed after optimize:')
    print(' ', executed)
    print('attention nodes:',
          [n['id'] for n in fused['nodes'] if n['op'] == 'attention'])
    removed = {n['id'] for n in spec['nodes']} - set(executed)
    print('eliminated chain nodes:', sorted(removed))
    seeds = {name: np.ones_like(value) for name, value in plain.items()}
    pg = graph.backward(spec, tape, seeds)
    fg = graph.backward(fused, ftape, seeds)

    fwd_err = max(float(np.max(np.abs(plain[name] - got[name])))
                  for name in plain)
    grad_err = max(float(np.max(np.abs(pg[name] - fg[name])))
                   for name in pg)
    print(f'max forward error: {fwd_err:.3e}')
    print(f'max input-gradient error: {grad_err:.3e}')
    print('trainable inputs returning gradients:', sorted(pg))
    return plain, got, pg, fg, fused


def main():
    # 1. Legacy model with a fully masked row and dynamic (trainable) scale.
    spec = model(shared=True, bias=True, dynamic=True)
    feeds = inputs(all_masked=True, scale=0.625)
    plain, got, pg, fg, fused = show('legacy DAG, fully masked row + '
                                     'dynamic scale', spec, feeds, 'float64')
    row = plain['out'][0, 1]
    print('fully masked output row (reference):', row.tolist())
    print('fully masked output row (fused):    ', got['out'][0, 1].tolist())
    print('probability side output row sums:',
          plain['prob'].sum(axis=-1).tolist())
    print('fixed_scale hint survived on fused node?:',
          any('fixed_scale' in n for n in fused['nodes']))
    print('scale gradient (reference vs fused):',
          float(pg['scale']), float(fg['scale']))

    # 2. New single block: broadcast bias, prob side output, both fusions.
    for scale_q in (False, True):
        spec = single_block_model(scale_q=scale_q)
        feeds = single_block_feeds(np.float64)
        plain, got, pg, fg, _ = show(
            f'new single block, scale-Q pattern={scale_q}',
            spec, feeds, 'float64')
        empty_rows = np.argwhere(~feeds['allow'].any(axis=-1))
        for b, l in empty_rows:
            assert np.all(plain['out'][b, l] == 0)
            assert np.all(plain['prob'][b, l] == 0)
        print(f'empty rows {empty_rows.tolist()} are zero in out and prob; '
              f'bias grad shape {pg["bias"].shape} matches feed shape')

    # 2b. Without side outputs the whole per-node chain must disappear.
    spec = single_block_model(prob_output=False)
    feeds = single_block_feeds(np.float64)
    fused = optimize(spec, feeds)
    print('\n=== fused graph without side outputs ===')
    print('executed nodes:', [n['id'] for n in fused['nodes']])
    assert all(n['op'] not in ('softmax_last', 'where_mask')
               for n in fused['nodes'])

    # 3. Two blocks sharing every trainable parameter.
    spec = two_block_model()
    feeds = two_block_feeds(np.float64)
    plain, got, pg, fg, fused = show('two blocks sharing parameters',
                                     spec, feeds, 'float64')
    print('attention count in fused graph:',
          sum(n['op'] == 'attention' for n in fused['nodes']))
    print('shared scale gradient accumulates both branches:',
          float(pg['scale']), 'vs', float(fg['scale']))

    # 4. float32 run with the tolerance required by the numeric contract.
    feeds32 = single_block_feeds(np.float32)
    spec = single_block_model()
    plain, tape = graph.evaluate(spec, feeds32)
    fused = optimize(spec, feeds32)
    got, ftape = graph.evaluate(fused, feeds32)
    seeds = {name: np.ones_like(value) for name, value in plain.items()}
    pg32 = graph.backward(spec, tape, seeds)
    fg32 = graph.backward(fused, ftape, seeds)
    ok = all(np.allclose(plain[n], got[n], atol=2e-5, rtol=2e-4)
             for n in plain)
    ok &= all(np.allclose(pg32[n], fg32[n], atol=2e-5, rtol=2e-4)
              for n in pg32)
    print('\nfloat32 fused-vs-reference within (2e-5, 2e-4):', ok)


if __name__ == '__main__':
    main()
