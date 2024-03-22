<div align="center">
<h1>Flower implementation of Oort client selection</h1>
<h3>Joseph Rance | Federated Learning final project</h3>
</div>

This codebase contains experiments to test a flower implementation of the Oort client selection algorithm. This is part of my Federated Learning module project. The original paper on Oort can be found at:

https://www.usenix.org/system/files/osdi21-lai.pdf

---

### Installation

To run this project, first modify lines 204 and 205 of `project/task/cifar10/dataset_preparation.py` to point to a suitable location to save the CIFAR10 dataset. Then run:
```bash
pip install .
bash run_scripts/download_cifar10.sh
bash run_scripts/launch.sh
```
The default configuration uses random sampling with $\sigma=2$. Oort can be activated by modifying line 198 of project/main.py.

### Structure

This codebase closely follows the structure of the [CaMLSys template](https://github.com/camlsys/fl-project-template).

The main additions are discussed in my technical report, and can be found in `project/task/cifar10` (CIFAR10 model implementation), `project/task/speech` (Google speech model implementation), `project/fed/server/stategy/strategy.py` (custom strategy for aggregating client utility values), and `project/fed/server/oort_client_manager.py` (Oort sampler implementation).