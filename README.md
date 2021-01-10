# CUDA-accelerated-Perlin-Noise-POC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4430682.svg)](https://doi.org/10.5281/zenodo.4430682)

## Installation
Code was written for Python 3.8.\
Install [PyTorch](https://pytorch.org/get-started/locally/).\
Install additional packages:
> pip install -r requirements.txt

## Usage
```
usage: cuda_accelerated_perlin_noise.py [-h] [-b BENCH] [-i] [-z ZOOM]

Generate image or benchmark

optional arguments:
  -h, --help            show this help message and exit
  -b BENCH, --bench BENCH, --benchmark BENCH
                        number of iterations (default 0)
  -i, --image           generate image
  -z ZOOM, --zoom ZOOM  zoom level (default 1)
```
