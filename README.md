# Readme

This repository contains code for: 

Brodbeck, C., Hannagan, T., & Magnuson, J.S. (2025). Recurrent neural networks as neuro-computational models of human speech recognition. 


## Contents

- `earshot` directory contains an installable module for estimating EARSHOT models
- `EARSHOT-Burgundy` contains scripts and notebooks (see https://jupytext.readthedocs.io) used for generating the figures in the paper
  - [FIG - paper.py](EARSHOT-Burgundy/FIG%20-%20paper.py) code for most components for the figures included in the paper
  - [A - analyze embedding.py](EARSHOT-Burgundy/A%20-%20analyze%20embedding.py) and [A - Cohort activation.py](EARSHOT-Burgundy/A%20-%20Cohort%20activation.py): additional figure components
  - [make predictors.py](EARSHOT-Burgundy/make%20predictors.py) and [make_cluster_predictors.py](EARSHOT-Burgundy/make_cluster_predictors.py): create predictors for MEG data
- `stimuli` contains a script used to generate the model input representations
- `tests` contains tests for certain functions in `earshot`


## Environment

Models were estimated using the environment in [env.yml](env.yml). 
