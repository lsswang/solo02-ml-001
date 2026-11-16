import json
import numpy as np
import graph
from examples import model, inputs
from optimizer import optimize

if __name__ == '__main__':
    m, feeds = model(), inputs()
    plain, tape = graph.evaluate(m, feeds)
    fused = optimize(m)
    got, ftape = graph.evaluate(fused, feeds)
    seed = {'out': np.ones_like(plain['out'])}
    pg, fg = graph.backward(m, tape, seed), graph.backward(fused, ftape, seed)
    print(json.dumps({'output':got['out'].tolist(),
                      'forward_max_error':float(np.max(np.abs(got['out']-plain['out']))),
                      'q_gradient_max_error':float(np.max(np.abs(pg['q']-fg['q']))),
                      'fused_nodes':sum(n['op']=='attention' for n in fused['nodes'])},ensure_ascii=False))
