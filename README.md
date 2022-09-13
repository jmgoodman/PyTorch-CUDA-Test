# PyTorch-CUDA-Test
 tests installation of Pytorch to ensure that GPU support is indeed up & running and meeting performance benchmarks

# Usage instructions
- Have Python 3.x and [PyTorch](https://pytorch.org/) installed.

- Two options are given: a Jupyter Notebook (`TestNotebook.ipynb`) and a simple Python script (`testscript.py`).

- To use `TestNotebook.ipynb`, it should be a simple matter of installing and running [Jupyter](https://jupyter.org/), navigating to where you cloned this repository, opening the notebook, and running it.

- To use `testscript.py`, navigate to where you cloned this repository and run `python -m testscript`

- In both cases, it's a good sign when "GPU is available" is printed.

- In both cases, two statements which indicate the amount of time taken (one for CPU, one nominally for GPU) are given. Great sign when GPU is taking less time than the CPU by an order of magnitude.