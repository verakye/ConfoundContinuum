# Confounder Control in Biomedicine Necessitates Conceptual Considerations Beyond Statistical Evaluations

## General information
This repository contains all the code needed to run the analyses for the paper "Confounder Control in Biomedicine Necessitates Conceptual Considerations Beyond Statistical Evaluations" (https://www.medrxiv.org/content/10.1101/2024.02.02.24302198v1.full).

### Data
This research has been conducted using data from UK Biobank, a major biomedical database (www.ukbiobank.ac.uk). Behavioural variables are derived from the UKB using the ukbb_parser in a modified version to be able to parse .tsv files (https://github.com/kaurao/ukbb_parser/tree/filetype_unknown). Most neuroimaging data are consumed using datalad (https://www.datalad.org). The data can not be made available here as it has constrained access. 

### Compute information
Most computations were run on a high throughput compute cluster with HTCondor scheduler. All scripts theoretically can be run on a single (modern) machine, but especially the feature extraction and prediction parts will run very long on a single machine.

## Repository structure
The repository follows the following folder structure:
1. `/data`: contains FC features, other features derived using DataLad
2. `/lib`: reusable code (functions, modules)
3. `/src`: scripts enumerated in consecutive order according to analysis
4. `/results`: output directory for scripts, follows the same ordering as `/src`

## Creating the environment:

1. Create a conda or mamba environment using the provided `requirements.yaml`:

```
mamba env create -f requirements.yaml 
```
(The ukbb_parser must be manually installed from the github branch listed above.)

2. After having set up the enviornment, to make the repository internal library structure available got o `./lib` (directory where the `setup.py`)  is located and run:
```
python setup.py develop
```

## Code explanations (`/src`)
In general, follow the respective numbering of subfolders and scripts within subfolders. If a script was executed on the cluster a `.submit` witht the same name as the to be executed python file exists. All code should be run in the root directory of the repository. Initial directories in `.submit` files will need to be adapted to indivual setups. 

1. feature extraction (`./src/1_feature_extraction/...`)
    - GMV
        1. generate submit and dag files e.g. `python ./src/1_feature_extraction/1_generate_submit_dag_gmd_Schaefer.py `
        2. submit dag: `condor_submit_dag -import_env ./src/1_feature_extraction/1_gmd_schaefer.dag` (and respectively for other atlases)
        3. merge single subject databases: e.g. `condor_submit ./src/1_feature_extraction/4_merge_gmd_SUIT_databases.submit` (and respectively for other atlases) 
    - FC: data from costum code from different project -> put FC.csv features in `./data/functional`. 
    - Convert .sqlite feature databases to .jay format for quicker IO: `python ./src/1_feature_extraction/7_convert_features2jay.py`
2. phenotyoe extraction (`./src/2_phenotype_extraction/...`)
    - get HGS target phenotypes and extract imaging subjects: either run `1_get_motor_phenotypes.py` or submit to cluster with .submit
    - clean the HGS target phenotypes: `2_clean_motor_phenotypes_IMG.py`
    - get all UKB phenotypes for stats CC (and extract imaging subjects):  run `3_get_possibleUKB_phenotypes.py` by submitting `3_get_possibleUKB_phenotypes.submit`
    - clean the UKB phenotypes: run `4_clean_possibleUKB_phenotypes.py`
    - Get TIV: `5_get_TIV.py`
3. statistical continuum (stats+visualization): (`./src/3_statistical_continuum/...`)
    - Calculate the correlation of GMV and HGS with potential confounders: `./3_statistical_continuum/1_ConfoundImportance_allUKB.py`
    - Visualize the stats CC: `2_Visualize_confoundImportance_allUKB.py`
4. predictions: (`./src/4_predictions/...`)
    - Create the pipeline options: `./4_prediction/1_create_pipeline_options.py`
    - Script that actually performs the predictions and makes a OOS prediction plot for each run in 5 different colours: `2_predict.py`
    - `2_predict.submit` -> submit file to run all jobs created in `./4_prediction/1_create_pipeline_options.py` with the prediction script `2_predict.py`
    - `3_merge_prediction_summaries`: Read in all single prediction summaries DFs and merge into one summary DF

