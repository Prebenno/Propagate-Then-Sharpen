# Propagate, then Sharpen

<img width="1418" height="471" alt="image" src="https://github.com/user-attachments/assets/5bafb559-0687-4287-b830-43a6747309a0" />

Code for the paper. It contains the method, baselines, the three frozen backbones (MLP, GCN, GraphSAGE), every experiment in the paper and the CSVs the tables and figures were made from. The main method is in `src/methods/pts.py`, it is under 100 lines: `graph_operator` builds S, `sharpen` is the reaction in eq. (3)–(4) and `pts` alternates the sharpening and APPNP-style propagation.

## Install

We used Python 3.12 on Linux on a HPC cluster. `pip install -r requirements.txt` covers the code itself, converting the CS-TAG graphs additionally needs `dgl` and `huggingface_hub`. The full run needs a GPU supporting CUDA.

## Most important files

- `src/methods/baselines.py` – APPNP, PPR-Prob, LAME-Graph, Correct & Smooth, Logit-Sharp and the row-normalised variants. `src/methods/graph_tv.py` – Graph-TV.
- `src/data.py` loading, symmetrisation and splits, `backbones.py` the MLP, GCN and GraphSAGE; `selection.py` the Optuna search and the Graph-TV grid,  `diagnostics.py` accuracy, calibration, energies, per-node statistics and runtime.
- `pretrained/` – MLP checkpoints, split indices and clean `Q`/`Z` for WikiCS, Cora-TAPE, PubMed-TAPE and TAPE-Arxiv23, splits 0–2, model seeds 3 and 4. These come from a later rerun, so they are not the exact checkpoints behind the tables and numbers will differ slightly.
- `results/paper/` – the CSV exports behind the tables and figures.
- `configs/paper.yaml` – the whole protocol in one place: datasets, splits, seeds, draws, severities, search spaces and budgets.

## Quick check

This runs PtS on saved clean predictions, so nothing is trained and no node features are downloaded:

```sh
python -m src.data --datasets wikics --root data
python main.py --pretrained --datasets wikics --backbones mlp --splits 1 --seeds 3 \
               --trials 10 --experiment main --output runs/demo
```

Tunes APPNP, PPR-Prob and PtS on the validation nodes with 10 Optuna trials each and writes one CSV per unit under `runs/demo/`, with test accuracy, NLL, Brier and ECE for every method and the selected hyperparameters. The paper used 250 trials, 10 is enough to see the pipeline run end to end and still gives good results in my experience.

## Reproduce the paper

Data first. `src/data.py` puts everything under `data/<dataset>/`:

- WikiCS, ogbn-arxiv, ogbn-products, Roman-Empire and Amazon-Ratings are downloaded by PyTorch Geometric / OGB on first use.
- Ele-Photo, Ele-Computers and Books-History come from the CS-TAG Hugging Face repository with `--cstag`, the published `*_roberta_base_512_cls.npy` features are used unchanged.
- Cora-TAPE, PubMed-TAPE and TAPE-Arxiv23 are in this repository as `data/<dataset>/graph.pt`, because TAPE distributes its data through Google Drive. TAPE-Arxiv23 is TAPE's own `arxiv_2023/graph.pt`.

Load data
```sh
python -m src.data --datasets all --root data --cstag
```
Then for the full run:
```sh
bash run_all.sh --device cuda
```

1. `train` – MLP, GCN and GraphSAGE for every dataset × split × model seed (828 units), saving checkpoints and clean `Q`/`Z`.
2. `predictions` – corrupted features through the frozen checkpoints, σ ∈ {0.5, 1, 1.5, 2}, three noise draws.
3. `main` – APPNP, PPR-Prob, PtS and Reaction OFF, 250 Optuna trials per unit, severity and draw (Tables 1, 2, 7, 10, 14).
4. `external`, `mass` – LAME-Graph, Graph-TV and the row-normalised variants (Tables 8, 9, 16).
5. `transfer`, `depth`, `energy`, `calibration`, `per_node` – data for Figures 3, 4, 6 and Tables 17, 18, 20.
6. `timing` – Table 19, on the checkpoints from step 1.

Each unit writes `<stage>.<access>.csv` under `runs/paper/<dataset>/<backbone>/split_<s>/seed_<k>/`, one row per severity, draw and method. Paper numbers average draws within a seed, seeds within a split, then splits; the reported SD is over splits, or over seed and draw for the single-split OGB graphs. Dataset means weight the nine main graphs equally and never include the two heterophilic controls. Differences between methods are taken within a unit, severity and draw before averaging.

We did not run this as one job. The stages were spread over several GPU and CPU nodes on a cluster and submitted separately, rerunning the same command skips every unit whose checkpoint, predictions or selections already exist. Almost all of the compute goes to Graph-TV, roughly 60% of the total even with less tuning: it is a full primal–dual solve per candidate λ for every unit, severity and draw. APPNP, PPR-Prob and PtS are quick by comparison, milliseconds per propagation step (Table 19), the `main` stage is long only because of the 250 Optuna trials per unit, severity and draw.

## Data sources

- WikiCS: https://github.com/pmernyei/wiki-cs
- Cora / PubMed / Arxiv23 (TAPE): https://github.com/XiaoxinHe/TAPE
- Ele-Photo / Ele-Computers / Books-History (CS-TAG): https://huggingface.co/datasets/Sherirto/CSTAG
- ogbn-arxiv / ogbn-products (OGB): https://ogb.stanford.edu/docs/nodeprop/
- Roman-Empire / Amazon-Ratings: https://github.com/yandex-research/heterophilous-graphs
