# Data module for calculating geothermal potential and costs.

This module calculates geothermal heat or power production based on a temperature model by Limberger et al. (2014).

A modular `snakemake` workflow built for [`clio`](https://clio.readthedocs.io/) data modules.

## Using this module

This module can be imported directly into any `snakemake` workflow.
Please consult the integration example in `tests/integration/Snakefile` for more information.

## Development

We use [`pixi`](https://pixi.sh/) as our package manager for development.
Once installed, run the following to clone this repo and install all dependencies.

```shell
git clone git@github.com:calliope-project/geo_module.git
cd geo_module
pixi install --all
```

For testing, simply run:

```shell
pixi run test-integration
```

To view the documentation locally, use:

```shell
pixi run serve-docs
```

To test a minimal example of a workflow using this module:

```shell
pixi shell    # activate this project's environment
cd tests/integration/  # navigate to the integration example
snakemake --use-conda --cores 2  # run the workflow!
```
