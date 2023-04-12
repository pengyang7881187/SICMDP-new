# SICMDP
Liangyu Zhang, Yang Peng, Wenhao Yang, Zhihua Zhang. Semi-Infinitely Constrained Markov Decision Processes and Efficient Reinforcement Learning

The repository includes the implementation of the environments and algorithms (SI-CPO and SI-CPPO) in the paper, see https://github.com/pengyang7881187/SICMDP for the implemntation of SI-CRL algorithm.

The tabular environment discharge of sewage and SI-CPO algorithm are included in *tabular_envs*, and other directories include the code for the ship route planning environment and SI-CPPO algorithm based on RLlib.

## Requirements
* [Gurobipy](https://www.gurobi.com/)
* [Gymnasium](https://gymnasium.farama.org/): ```pip install gymnasium```
* [Pytorch](https://pytorch.org/): ```pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117```
* [RLlib](https://www.ray.io/): ```pip install "ray[rllib]"```
