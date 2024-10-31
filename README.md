# Morphing lattices

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=ecf0f1&labelColor=34495e)
[![](https://img.shields.io/badge/Paper-10.1002/adma.202310743-blue?logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.1002/adma.202310743)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281/zenodo.10499196-blue?logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.5281/zenodo.10499196)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbertoldi-collab%2Fmorphing-lattices&count_bg=%2327AE60&title_bg=%2334495E&icon=github.svg&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

Simulation and design of shape-morphing LCE lattices

<p float="center">
  <img src="inverse_design/circle_4_pointed_star_spikeness_0.10_n1_16_n2_18/pareto/weights_0.93_0.07_best/discrete_animation_with_targets_temperature.gif" width="48%" />
  <img src="inverse_design/curvy_square_spikeness_0.10_n1_16_n2_18/pareto/weights_0.43_0.57_best/discrete_animation_with_targets_temperature.gif" width="48%" />
</p>

## Paper

This repository contains all the code developed for the paper:
[A. Kotikian, A. A. Watkins, G. Bordiga, A. Spielberg, Z. S. Davidson, K. Bertoldi, J. A. Lewis, Liquid Crystal Elastomer Lattices with Thermally Programmable Deformation via Multi-Material 3D Printing. _Adv. Mater._ 2024, 2310743.](https://doi.org/10.1002/adma.202310743)

## Installation

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/morphing-lattices.git@main
```

## Dev notes

<details>
<summary><b>Other ways to install</b></summary>

### Installation (with poetry)

The dependency management of the project is done via [poetry](https://python-poetry.org/docs/).

To get started

- Install [poetry](https://python-poetry.org/docs/)
- Clone the repo
- `cd` into the root directory and run `poetry install`. This will create the poetry environment.
- If you are using vscode, search for `venv path` in the settings and paste `~/.cache/pypoetry/virtualenvs` in the `venv path` field. Then select the poetry enviroment as python enviroment for the project.

### Installation (with pip)

This is meant to be a quick way to use the package.
It is not recommended to use this method for development.

To get started

- Clone the repo
- `cd` into the root directory and run `pip install -e .`

</details>

## How to cite

If you use this code in your research, please cite the related [paper](https://doi.org/10.1002/adma.202310743):

```bibtex
@article{kotikian_2024,
    title   = {Liquid Crystal Elastomer Lattices with Thermally Programmable Deformation via Multi-Material {{3D}} Printing},
    author  = {Kotikian, Arda and Watkins, Audrey A. and Bordiga, Giovanni and Spielberg, Andrew and Davidson, Zoey S. and Bertoldi, Katia and Lewis, Jennifer A.},
    year    = {2024},
    journal = {Advanced Materials},
    pages   = {2310743},
    issn    = {1521-4095},
    doi     = {10.1002/adma.202310743},
}
```
