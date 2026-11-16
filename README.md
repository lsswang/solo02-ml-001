# Small differentiable attention graph

Install `requirements.txt` in Python 3.11, then run `python demo.py`.

`attention.py` is the original fixed-tree model's numerical primitive. `graph.py`
is a small tensor-DAG interpreter with reverse-mode differentiation. The legacy
`optimizer.py` accepts the original single-consumer tree exported by `examples.model()`;
it is not yet a general DAG optimizer. `examples.py` also constructs the new
model shapes used by the requested extension. Inputs are generated locally.

Graph nodes are topologically ordered dictionaries containing `id`, `op` and
`inputs`; input-node names address NumPy arrays in the feed dictionary. The
`outputs` list names observable nodes. `evaluate` returns output arrays plus a
per-execution tape, and `backward` accepts output cotangents. The original graph
has more general scalar-input derivative semantics than the old fused node,
whose scale was copied from an exporter hint.

No model weights, network services, GPUs or external datasets are required.
