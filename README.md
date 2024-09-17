# Bi-MIChI
**Bi-Capacity Choquet Integral for Sensor Fusion with Label Uncertainty**  

_Hersh Vakharia and Xiaoxiao Du_  

Accepted to 2024 FUZZ-IEE, Presented at 2024 WCCI in Yokohama, Japan

[[`arXiv`](https://arxiv.org/abs/2409.03212)] [[`IEEEXplore`](https://ieeexplore.ieee.org/document/10611865)]

## Installation Prerequisites
This code was tested using Python 3.10.  
Install requirements with `pip3 install -r requirements.txt`.

## Demo
The demo reproduces the "UM" experiement from the paper via the jupyter notebook [`um_test.ipynb`](um_test.ipynb). Simply run the notebook to see the results.

## Code Structure
Here is a brief summary of the code structure. For detailed usage, see the docustrings in the code, as well as the demo jupyter notebook.

[`bicap.py`](bicap.py):
- `Bicapacity` (class): defines the structure and usage of a bicapacity 
- `BicapacityGenerator` (class): random generation of new bicapacities
  - _useZeroBound_ param determines whether bicapacity zero bound is enforced (Obj 1 vs Obj 2)
- `choquet_integral` (function): given data and a bicapacity, computes and returns the choquet integral.

[`bicap_train.py`](bicap_train.py):
- `BicapEvolutionaryTrain` (class): Contains code for the optimization framework, given data, labels, and params 
  - Params dictionary example:
    ```python
    param = {
      "max_iter": 5000, # maximum optimization iterations
      "eta": 0.8, # small-scale mutation rate
      "pop_size": 8, # bicap population size
      "fitness_thresh": 0.001, # fitness threshold for stopping condition
      "use_zero_bound": True # enforce zero bound or not
    }
    ```
[`utils.py`](utils.py): Contains miscellaneous utility functions used throughout the rest of the code.

## License
This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2024 H. Vakharia and X. Du. All rights reserved.

## Citing Bi-MIChI
If you use the Bi-MIChI fusion framework, please cite the following reference using the following BibTeX entries.  
```
@INPROCEEDINGS{10611865,
  author={Vakharia, Hersh and Du, Xiaoxiao},
  booktitle={2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)}, 
  title={Bi-Capacity Choquet Integral for Sensor Fusion with Label Uncertainty}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  keywords={Training;Uncertainty;Limiting;Soft sensors;Measurement uncertainty;Data integration;Object detection;bi-capacity;choquet integral;fuzzy measures;sensor fusion;label uncertainty;classification},
  doi={10.1109/FUZZ-IEEE60900.2024.10611865}}
```

## Related Work
Multiple Instance Choquet Integral (MICI) [[`arXiv`](https://arxiv.org/abs/1803.04048)] [[`Code Repo`](https://github.com/GatorSense/MICI)]  

Multiple Instance Multi-Resolution Fusion (MIMRF) [[`arXiv`](https://arxiv.org/abs/1805.00930)] [[`Code Repo`](https://github.com/GatorSense/MIMRF)]  

MIMRF with Binary Fuzzy Measures (MIMRF-BFM)  [[`arXiv`](https://arxiv.org/abs/2402.05045)] [[`Code Repo`](https://github.com/hvak/MIMRF-BFM)] 
