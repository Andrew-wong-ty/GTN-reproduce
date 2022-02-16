# GTN-reproduce
This repository is the reproduce of [Graph Transformer Networks(GTN)](https://arxiv.org/abs/1911.06455).

## How to run?
open ```reproduce.ipynb```, then execute commands in the notebook one by one
- CUDA is required

## Experimental details
Experiments are carried out on a single NVIDIA A100 SXM4. Hyperparameter settings:
- Train epochs: 40
- Learning-rate: 0.005
- weight decay: 0.001
- Seed: 16
## Reproduce Results
Table : Evaluation results on the node classification task of different datasets (F1 score)
|            | result in paper    |  reproduce result  |
| :-:   | :-:   | :-: |
| DBLP        | 94.18      |   94.02    |
| ACM        | 92.68      |   92.12    |
| IMDB        | 60.92      |   58.75    |
