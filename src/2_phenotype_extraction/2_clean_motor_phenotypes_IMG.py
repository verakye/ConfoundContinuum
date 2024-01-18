# %%
# imports
import os
from pathlib import Path

import pandas as pd

from confoundcontinuum.logging import logger
from confoundcontinuum import targets

# %%
# directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())

# input
phenotype_dir = project_dir / 'results' / '2_phenotype_extraction'
IMG_fname = phenotype_dir / '11_HGS_exICD10-V-VI-stroke_imgSbjs.csv'

# %%
# general definitions

# cleaned output -> see fnames in respective section

# cleaning
session2keep = 'ses-2'
min_nonans = 25000

# outlier removal
std_limit = 4
include_ageOL = True

# %%
# Clean and prepare IMG subjects' phenotypes (used as targets)

# 0) Load IMG subjects
healthy_HGS_img = pd.read_csv(IMG_fname.as_posix())
logger.info('Loaded the healthy imaging dataset.')

# 1) Clean and select session
healthy_HGS_img_clean = targets.shape_phenotypes_ukbb(
    healthy_HGS_img, session2keep=session2keep)
logger.info(
    'Shaped the imaging dataset to remove irrelevant general columns and chose '
    f'{session2keep =}. SubjectID was set as index.')

# 2) Keep only relevant columns
general_cols = ['Age', 'Sex']
filter_col = [col for col in healthy_HGS_img_clean if col.startswith('Hand')]
HGS = healthy_HGS_img_clean[filter_col + general_cols].copy()

# 3) category specific column cleaning
HGS = targets.modify_hgs_ukbb(HGS)

# 3) Remove NaN rows per category (or columns if <25k non_NaN rows)
HGS_noNaN = targets.dropnan_ukbb(HGS, min_nonans=min_nonans)
logger.info(
    'Dropped rows with NaN in any column. Removed '
    f' a column if it had less then {min_nonans = } rows.')

# 4) Remove outliers
HGS_noNaN_noOL, _, _, _ = targets.remove_outliers_ukbb(
            HGS_noNaN, std_limit=std_limit, non_number_cols=['Sex'],
            drop_OL=True,
            include_ageOL=include_ageOL,
            )
logger.info(
    f'Outliers were removed with {include_ageOL = } and criteria '
    f'{std_limit = }.')

# 5) Check sex balance per category/variable
HGS_sex_balance = targets.check_sex_balance_ukbb(HGS_noNaN_noOL)

# 6) Save
HGS_noNaN_noOL.to_csv(
    phenotype_dir / '20_HGS_exICD10-V-VI-stroke_IMG_noNaN-noOL.csv')

# %%
