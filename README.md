# Wrapper for Drake's kinematic trajectory optimization
Simple wrapper around Drake's kinematic trajectory optimization

## Installation

- Clone the repository and `cd` into it.
  ```bash
  git clone https://github.com/siddancha/drake-trajopt.git
  cd drake-trajopt
  ```

- Create a new virtual environment.
  ```bash
  python -m venv .venv --prompt=drake-trajopt
  source .venv/bin/activate
  ```

### Installation via `pip`

- Run the following command:
  ```bash
  pip install -e .
  ```

### Installation via `pdm`

- Install `pdm`.
  ```bash
  pip install pdm
  ```

- Run the following command:
  ```bash
  pdm install
  ```

## Example usage

- Run the example scripts found in the `examples` directory.
  ```bash
  python examples/spot_and_toolbox.py
  ```
This should open a Meshcat window in your browser (at http://localhost:7000/).

<img src="media/trajopt.gif" width="50%">

This example contains a Spot robot and an articulated toolbox object loaded from a URDF file.
The initial path asks the robot to move in a straight line colliding with the toolbox.
This is shown in the white line.

Trajopt modifies this trajectory to avoid collisions while minimizing the length of the trajectory to the goal.

> [!NOTE]
> This is a particularly hard example because the initial (white) path is in collision,
> taking the SNOPT solver many iterations to converge.
> Paths output by an RRT, for example, will be collision-free to begin with.
> In those cases, trajectory optimization will be able to smooth the path more efficiently.
