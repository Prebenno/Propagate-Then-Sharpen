# Propagate, then Sharpen
Code for training the backbones and running PtS, the baselines and the paper's experiments.

## Quick start
Use python 3.12 then run these commands

python -m pip install -r requirements.txt
python data.py --datasets wikics --root data
python reproduce.py --datasets wikics --backbones mlp --splits 1 --seeds 1 --draws 1 --sigmas 0 2 --epochs 20 --trials 10 --experiment main --output runs/quick


This runs a small WikiCS example on CPU with reduced training and tuning budgets. Results are saved in runs/quick/ , with summary CSVs in runs/quick/report/tables/.
## Full run

To run the full board, you need all datasets, then run this:
```sh
python reproduce.py --data-root data --output runs/paper --device cuda --threads 8(optional)
```

This trains MLP, GCN and GraphSAGE and runs all experiments. Use --datasets wikics to run one dataset, or --experiment main for the full comparisons. Use `--device cpu` to run on CPU (not recommended). The full run is expensive, especially on the OGB graphs. Existing checkpoints and predictions are reused.

## Data
`data.py` can download `wikics`, `ogbn-arxiv`, `ogbn-products`, `roman-empire` and `amazon-ratings`.
Other datasets need `data/<dataset>/graph.pt` containing `x`, `y` and `edge_index`. For CS-TAG, install `dgl` and `huggingface_hub`, then run:
python data.py --datasets ele-photo ele-computers books-history --root data --cstag
The tape datasets can be downloaded from here: https://github.com/XiaoxinHe/TAPE

