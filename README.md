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
