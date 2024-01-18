# %%
# imports
import os
from pathlib import Path
import tempfile

import pandas as pd

from confoundcontinuum.logging import logger, raise_error

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402
# %%
# set params

# input
feature = 'all_gmv'
shuffle_feature = False

# %%
# directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
base_dir = project_dir / 'results'
phenotype_dir = base_dir / '2_phenotype_extraction'

REPO_URL = 'ria+http://ukb.ds.inm7.de#~cat_m0wp1'

# fnames
tiv_in_name = 'cat_rois_Schaefer2018_600Parcels_17Networks_order.csv'
tiv_out_fname = (
    phenotype_dir / '50_TIV.csv'
    )

# %%
# Get VBM datalad dataset to temp directory to get TIV

with tempfile.TemporaryDirectory() as tmpdir:
    # Define tmp subdirectories
    tmp_db_path = Path(tmpdir) / 'data'
    tmp_cat_inst_dir = tmp_db_path / 'cat_m0wp1'
    tmp_tiv_fname = tmp_cat_inst_dir / 'stats' / tiv_in_name

    # Create the directories if they do not exist
    tmp_cat_inst_dir.mkdir(exist_ok=True, parents=True)

    # Clone VBM datalad dataset
    logger.info(
        f'Cloning VBM datalad dataset from {REPO_URL} to {tmp_cat_inst_dir}')
    dl.install(path=tmp_cat_inst_dir, source=REPO_URL)  # type: ignore
    logger.info('Dataset cloned.')

    # check if symbolic link of cloned file exists
    if not tmp_tiv_fname.is_symlink():
        raise_error(f'TIV file {tmp_tiv_fname} does not exist.')

    # Get TIV file
    logger.info('Get TIV file.')
    dl.get(path=tmp_tiv_fname, dataset=tmp_cat_inst_dir)
    logger.info('Got TIV file.')

    # load VBM as nifti, resolution 1.5 mm
    logger.info(f'Loading TIV file from {tmp_tiv_fname}')
    TIV = pd.read_csv(
        tmp_tiv_fname, usecols=['SubjectID', 'Session', 'TIV'],
        index_col=['SubjectID', 'Session'])
    TIV = TIV.xs('ses-2', level=1, drop_level=True).copy()
    logger.info('TIV loaded.')

    # save TIV to .csv
    TIV.to_csv(tiv_out_fname)
    logger.info(f'Saved TIV to {tiv_out_fname}')
