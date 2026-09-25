# Propagate, then Sharpen
This repo contains the code nessesary to reproduce the results of the paper. It runs post hoc refinement of frozen node predictions and containts the three backbones, all experiments in the paper and the CSVs used for the tables in the paper.

The method itself is contained in `src/methods/pts.py `


## Install

To run this you need Python 3.12, this was ran on a cluster using linux and Python, first run `pip install -r requirements.txt` to get the nessesary packages needed to run it, it is highly recommended to use a GPU supporting CUDA to run the full run.


## Important files

- `src/methods/pts.py` – the method. 
   `graph_operator` builds S, 
   `sharpen` is the reaction in eq. (3)–(4), and 
   `pts` alternates propagation and sharpening.
   `baselines.py` has APPNP, PPR-Prob, LAME-Graph and Correct & Smooth, 
   `graph_tv.py` has Graph-TV.


- `src/` – `data.py` (loading, symmetrisation, splits), 
   `backbones.py` (MLP, GCN, GraphSAGE),
  `selection.py` (Optuna search and the Graph-TV grid), 
  `seeds.py` (how the random streams are derived from dataset/split/seed/draw), 
  `diagnostics.py` (accuracy, calibration, energies, per-node strata, timing) and 
  `plotting.py`. (Plots for model)

  - `pretrained/` – 28 MLP checkpoints with their split indices and clean `Q`/`Z` (`s0_d0.pt`):
  WikiCS, Cora-TAPE, PubMed-TAPE and TAPE-Arxiv23 for splits 0–2, and
  ogbn-products on the official split, all with model seeds 3 and 4. These are from a later rerun made for 
  this deliverable

- `results/paper/` – the CSV exports the tables and figures were made from.

- `configs/paper.yaml` – the full protocol: datasets, splits, seeds, draws, severities, search spaces and budgets.

## Tests

We have some clean predictions saved for a subset of the data, so PtS can be run and tested without training anything or downloading the graph features, this can be ran with 
```sh
python -m src.data --datasets wikics --root data          #graph and splits only
python pretrained.py                                       #lists the saved checkpoints and Q/Z files
python main.py --pretrained --datasets wikics --backbones mlp --splits 1 --seeds 3 \
               --trials 10 --experiment main --output runs/demo
```
This loads the saved clean logits `Z` and probabilities `Q` for split 1 / seed 3, tunes
APPNP, PPR-Prob and PtS on the validation nodes with 10 Optuna trials each, and writes
test accuracies and the selected hyperparameters to `runs/demo/report/`. The paper uses (`--trials 250`), which might take a little longer.

## Reproduce the paper

`src/data.py` downloads and converts the datasets from the sources listed at the bottom
(WikiCS, TAPE, CS-TAG, OGB and the heterophilous benchmark) into the layout the code expects:


```sh
bash run_all.sh --dry-run                                        # prints the stages, runs nothing
bash run_all.sh --data-source ../paper-data.zip --device cuda    # full run
bash run_all.sh --data-source ../paper-data.zip --device cuda --resume   # continue an interrupted run
```

The stages run in this order:

1. `train` – MLP, GCN and GraphSAGE for every dataset × split × model seed (828 units in total),
   saving checkpoints and clean `Q`/`Z`.
2. `predictions` – corrupted features through the same frozen checkpoints, σ ∈ {0.5, 1, 1.5, 2},
   three noise draws.
3. `main` – APPNP, PPR-Prob, PtS and Reaction OFF, 250 Optuna trials per unit, severity and
   draw (Tables 1, 2, 7, 10, 14).
4. `external`, `mass` – LAME-Graph, Graph-TV and the row-normalised variants (Tables 8, 9, 16).
5. `transfer`, `depth`, `energy`, `calibration`, `per_node` – Figures 3, 4, 6 and Tables 17, 18, 20.
6. `timing` – Table 19, on the checkpoints from step 1.
7. `report` – CSVs and plots in `runs/paper/report/`, plus the two-community example (Figure 5).


The full run was done paralell over multiple GPUs and CPUs, graphTV is extremley heavy to run and accounts for rougly 60% of the total time.


## Data sources

WikiCS (https://github.com/pmernyei/wiki-cs), 
Cora/PubMed/Arxiv23 from TAPE (https://github.com/XiaoxinHe/TAPE), 
Ele-Photo/Ele-Computers/Books-History from CS-TAG (https://huggingface.co/datasets/Sherirto/CSTAG),
ogbn-arxiv/ogbn-products from OGB (https://ogb.stanford.edu/docs/nodeprop/)
 Roman-Empire/Amazon-Ratings from https://github.com/yandex-research/heterophilous-graphs. 

