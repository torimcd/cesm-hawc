# Installation

cesm-hawc requires **Python 3.11 or later**.

## From PyPI

```bash
pip install cesm-hawc          # base tier: column extraction and save-inputs
pip install cesm-hawc[sim]     # + forward model and L2 retrieval
```

% TODO: the [sim] extra currently pins aliprocessing to a git commit, which
% PyPI does not accept. Update this page once that dependency is resolved.

## From source

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
micromamba env create -f environment.yml
micromamba activate hawc_env
pip install -e ".[sim,dev]"
```

`environment.yml` installs `sasktran2` and the scientific stack from
conda-forge, plus `hawcsimulator` and the pinned `aliprocessing` commit
from pip.

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
