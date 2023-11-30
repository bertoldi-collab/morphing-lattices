# Morphing lattices

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=ecf0f1&labelColor=34495e)

Simulation and design of shape-morphing LCE lattices

<p float="center">
  <img src="inverse_design/circle_4_pointed_star_spikeness_0.10_n1_16_n2_18/pareto/weights_0.93_0.07_best/discrete_animation_with_targets_temperature.gif" width="48%" />
  <img src="inverse_design/curvy_square_spikeness_0.10_n1_16_n2_18/pareto/weights_0.43_0.57_best/discrete_animation_with_targets_temperature.gif" width="48%" />
</p>

## Installation

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/morphing-lattices.git@main
```

## Dev notes

<details>
<summary>
No experience with Python and/or git? Open this :point_down:
</summary>

Go ahead with the following steps and you will learn along the way.

- Install [git](https://git-scm.com/downloads) and [learn a bit about it](https://youtu.be/RGOj5yH7evk).
- If you don't have a favorite IDE yet, install [vscode](https://code.visualstudio.com/) (this is just an environment that helps you writing code as well as managing git repositories).
- Install [Python](https://www.python.org/downloads/windows/) (any version above 3.7 should work fine).
- Follow the rest of the installation instructions below.

</details>

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
