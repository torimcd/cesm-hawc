# Installation

cesm-hawc requires **Python 3.11 or later**.

## Recommended: conda environment

To run the simulator (`cesm-hawc run`, the `[sim]` tier), install cesm-hawc
into the conda environment defined by the project's `environment.yml`. It
installs `sasktran2` and the scientific stack from conda-forge, and
`hawcsimulator` and `aliprocessing` from PyPI, in versions that work
together.

```bash
curl -O https://raw.githubusercontent.com/torimcd/cesm-hawc/main/environment.yml
conda env create -f environment.yml
conda activate hawc_env
pip install "cesm-hawc[sim]"
```

`mamba` and `micromamba` accept the same commands in place of `conda`.

```{tip}
The conda environment is the tested setup for the `[sim]` tier. It keeps
every compiled package on conda-forge's builds, which avoids conflicts
between copies of the OpenMP library bundled in pip wheels.
```

## Base tier only

Extracting WACCM columns and saving simulator-ready inputs
(`cesm-hawc save-inputs`) needs no simulator, and installs with pip in any
Python 3.11+ environment:

```bash
pip install cesm-hawc
```

## From source

For development, clone the repository and install it in editable mode into
the same conda environment:

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
conda env create -f environment.yml
conda activate hawc_env
pip install -e ".[sim,dev]"
```

See [Development](reference/development.md) for running the tests and
building these docs.

## Checking the install

```bash
cesm-hawc --help
python -c "import cesm_hawc; print(cesm_hawc.__version__)"
```

For the `[sim]` tier, also check that the simulator imports:

```bash
python -c "import sasktran2, hawcsimulator; print('sim tier OK')"
```

## On first use

The `[sim]` tier builds several Mie scattering databases the first time it
runs, and caches them on disk. Expect the first simulation to take noticeably
longer than later ones. The CLI builds these caches once in the main process
before starting worker processes.

## HPC clusters

See [Running on HPC](user-guide/hpc.md) for the Alliance Canada setup script
and the SLURM job script.
