# Motion and Color-change Magnification with Riesz Pyramids
This repository contains the code for the GUI presented in our paper entitled [Simultaneous magnification of subtle motions and color variations in videos using Riesz pyramids](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001795).

## Installation

### Installing with pip
To install the project and its dependencies you can run
```bash
pip install .
```

If you plan on editing the code, it might be useful to install it using
```bash
pip install -e .[dev]
```

### Conda environment.yml
A conda environment file is also available. To use it, run

```bash
conda env create --file environment.yml
conda activate riesz-env
pip install .
```

## Usage
To start the GUI, run 
```bash
riesz_simul_gui --video <path-to-video>
```
More options are available in `riesz_simul_gui --help`.

When the preprocessing has finished, press `p` to play (or pause) the video and press over it to update the chrominance-based mask.
In the beginning, nothing is selected and therefore no magnification will be displayed.

The following table shows the keyboard keys along with their respective functions

| Key | Function|
|-----|---------|
| p | play/pause|
| q | quit |
| t | update temporal filter |

A video showing examples and a demo of the application is available [here](https://www.youtube.com/watch?v=qTBCHFejj-4).

## Citation
If you find this code useful, consider citing the paper:

```bibtex
@article{fontanari_simultaneous_2021,
	author = {Fontanari, Thomas V. and Oliveira, Manuel M.},
	title = {Simultaneous magnification of subtle motions and color variations in videos using {Riesz} pyramids},
    journal = {Computers \& Graphics},
    year = {2021},
	volume = {101},
    pages = {35--45},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S0097849321001795},
	doi = {10.1016/j.cag.2021.08.015},
}
```