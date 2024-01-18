# %%
# imports
import os
from pathlib import Path
import tempfile

import pandas as pd

from confoundcontinuum.logging import logger
from confoundcontinuum import targets

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

# %%
# directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
# input
REPO_URL = "ria+http://ukb.ds.inm7.de#~super"
pheno_filename = 'ukb670018.tsv'

# output
phenotype_out_dir = project_dir / 'results' / '2_phenotype_extraction'
phenotype_out_dir.mkdir(exist_ok=True, parents=True)
logger.info(f'Output directory set to {phenotype_out_dir}.')

# %%
# general definitions

# parsing options
excludeICD10 = ["F", "G", "I60-I69"]
tobeparsed_hdrs = [53, 54, 21862]  # date & place assessment centre, time scan
tobeparsed_cats = [
    1039, 136, 130, 1008, 100090, 116, 113, 76, 100062, 100033, 100034, 100059,
    100070, 100069, 100006, 100026]  # "all" possible UKB

# parsing filenames
phenotype_parse_fname = (
    phenotype_out_dir / '30_allUKB_exICD10-V-VI-stroke_parsed')
phenotype_parsed_fname = (
    phenotype_out_dir / '30_allUKB_exICD10-V-VI-stroke_parsed.csv')

# imaging/non-imaging split
IMG_fname = (
    phenotype_out_dir / '31_allUKB_exICD10-V-VI-stroke_imgSbjs.csv')
nonIMG_fname = (
    phenotype_out_dir / '31_allUKB_exICD10-V-VI-stroke_nonimgSbjs.csv')

# %%
# ----------------------------- RUN ONCE --------------------------------------#

# 0) Get UKB super dataset
with tempfile.TemporaryDirectory() as tmpdir:
    # Directories
    tmpdir = Path(tmpdir)
    install_dir = tmpdir / 'ukb_super'
    phenotype_in_fname = install_dir / pheno_filename

    # Install and get UKBB .tsv from datalad super dataset
    logger.info(
        f'Create temporary directory {tmpdir} to get ukb super dataset')
    logger.info(f'Clone ukb_super datalad dataset from {REPO_URL}')
    dl.install(path=install_dir, source=REPO_URL)
    logger.info(
        f'Get phenotype tsv-file {phenotype_in_fname} from ukb_super dataset '
        f'in {install_dir}.')
    dl.get(path=phenotype_in_fname, dataset=install_dir)
    logger.info(
        f'Got phenotype tsv-file {phenotype_in_fname}.')

    # 1) Parse ALL phenotypes
    parse_process_result = targets.parse_phenotypes_ukbb(
        phenotype_in_fname, phenotype_parse_fname, excludeICD10=excludeICD10,
        inhdrs=tobeparsed_hdrs, incats=tobeparsed_cats)
    logger.debug(parse_process_result)

# %%
# 2) Separate imaginge and non-imagine subjects

# Load all parsed phenotypes
healthy_physiology_all = pd.read_csv(phenotype_parsed_fname.as_posix())

# get imaging subjects only (== all subjects with entry in field 53-2.0)
healthy_physiology_img = healthy_physiology_all[
    healthy_physiology_all[
        "Date_of_attending_assessment_centre-2.0"].notnull()].copy()
# Save imaging subjects
healthy_physiology_img.to_csv(IMG_fname.as_posix())

# get non-imaging subjects only (== all subjects NO entry in field 53-2.0)
healthy_physiology_nonimg = healthy_physiology_all[
    healthy_physiology_all[
        "Date_of_attending_assessment_centre-2.0"].isnull()].copy()
# Save non-imagine subjects
healthy_physiology_nonimg.to_csv(nonIMG_fname.as_posix())

# 3) Check distinctness of img-non-img subject dfs
img_subs = healthy_physiology_img.index.to_list()
nonimg_subs = healthy_physiology_nonimg.index.to_list()
INI = set(img_subs)-set(nonimg_subs)
NII = set(nonimg_subs)-set(img_subs)
if len(INI) == len(img_subs):
    logger.debug(
        'Great! The number of distinct subjects between the imaging and '
        'non-imaging dataframes is equal to the number of img-subjects '
        f'({len(img_subs)}).')
else:
    logger.debug(
        'Not good! There are some subjects in the non-imaging dataset which '
        'are also in the imaging dataset! This will lead to data leakage. '
        f'({len(img_subs)-len(INI)}).')
if len(NII) == len(nonimg_subs):
    logger.debug(
        'Great! The number of distinct subjects between the non-imaging and '
        'imaging dataframes is equal to the number of non-img-subjects '
        f'({len(nonimg_subs)}).')
else:
    logger.debug(
        'Not good! There are some subjects in the imaging dataset which '
        'are also in the non-imaging dataset! This will lead to data leakage. '
        f'({len(nonimg_subs)-len(NII)}).')

# ------------------------- END RUN ONCE --------------------------------------#

# %%
