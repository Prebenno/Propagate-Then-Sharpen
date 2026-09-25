# Propagate, then Sharpen

<img width="1418" height="471" alt="image" src="https://github.com/user-attachments/assets/5bafb559-0687-4287-b830-43a6747309a0" />

Code to reproduce the paper: post-hoc refinement of frozen node predictions, the three backbones, every experiment, and the CSVs behind the tables.

The method is in `src/methods/pts.py`.

## Install

Python 3.12. `pip install -r requirements.txt`. The full run needs a CUDA GPU.

## Files

- `src/methods/pts.py` – the method. `graph_operator` builds S, `sharpen` is the reaction in eq. (3)–(4), `pts` alternates propagation and sharpening.
- `src/methods/baselines.py` – APPNP, PPR-Prob, LAME-Graph and Correct & Smooth. `src/methods/graph_tv.py` – Graph-TV.
- `src/` – `data.py` (loading, symmetrisation, splits), `backbones.py` (MLP, GCN, GraphSAGE), `selection.py` (Optuna search and the Graph-TV grid), `seeds.py` (random streams from dataset/split/seed/draw), `diagnostics.py` (accuracy, calibration, energies, per-node strata, timing), `plotting.py`.
- `pretrained/` – MLP checkpoints with split indices and clean `Q`/`Z` for WikiCS, Cora-TAPE, PubMed-TAPE and TAPE-Arxiv23 (splits 0–2) and ogbn-products (official split), model seeds 3 and 4. From a later rerun.
- `results/paper/` – the CSVs the tables and figures were made from.
- `configs/paper.yaml` – the full protocol: datasets, splits, seeds, draws, severities, search spaces and budgets.

## Quick check

Runs PtS on saved clean predictions, without training or downloading node features:

```sh
python -m src.data --datasets wikics --root data
python main.py --pretrained --datasets wikics --backbones mlp --splits 1 --seeds 3 \
               --trials 10 --experiment main --output runs/demo
```

Tunes APPNP, PPR-Prob and PtS on the validation nodes with 10 Optuna trials each and writes test accuracies and selected hyperparameters to `runs/demo/report/`. The paper uses `--trials 250`.

## Reproduce the paper

```sh
bash run_all.sh --data-source ../paper-data.zip --device cuda
```

`src/data.py` downloads the datasets from the sources below and converts them to the layout the code expects. Stages, in order:

1. `train` – MLP, GCN and GraphSAGE for every dataset × split × model seed (828 units), saving checkpoints and clean `Q`/`Z`.
2. `predictions` – corrupted features through the frozen checkpoints, σ ∈ {0.5, 1, 1.5, 2}, three noise draws.
3. `main` – APPNP, PPR-Prob, PtS and Reaction OFF, 250 Optuna trials per unit, severity and draw (Tables 1, 2, 7, 10, 14).
4. `external`, `mass` – LAME-Graph, Graph-TV and the row-normalised variants (Tables 8, 9, 16).
5. `transfer`, `depth`, `energy`, `calibration`, `per_node` – Figures 3, 4, 6 and Tables 17, 18, 20.
6. `timing` – Table 19, on the checkpoints from step 1.
7. `report` – CSVs and plots in `runs/paper/report/`, plus the two-community example (Figure 5).

The full run was parallelised over several GPUs and CPUs. Graph-TV alone is about 60% of the total time.

## Data sources

- WikiCS: https://github.com/pmernyei/wiki-cs
- Cora / PubMed / Arxiv23 (TAPE): https://github.com/XiaoxinHe/TAPE
- Ele-Photo / Ele-Computers / Books-History (CS-TAG): https://huggingface.co/datasets/Sherirto/CSTAG
- ogbn-arxiv / ogbn-products (OGB): https://ogb.stanford.edu/docs/nodeprop/
- Roman-Empire / Amazon-Ratings: https://github.com/yandex-research/heterophilous-graphs
