# %%
# imports
from pathlib import Path
import tempfile
import time
import os

import pandas as pd

from confoundcontinuum.logging import configure_logging, log_versions, logger, \
    raise_error
from confoundcontinuum.io import save_features
from confoundcontinuum.features import ukbb_get_searched_categories_description
from confoundcontinuum.features import ukbb_get_category_datafields

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

# %%
# configurations

configure_logging()
log_versions()

# %%
# Basic definitions

# directories
# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
logger.info(f'Project directory was set to {project_dir}.')
# input
REPO_URL = "ria+http://ukb.ds.inm7.de#~super"
FS_filename = 'ukb670018.tsv'
# output
surface_feature_dir = (
    project_dir / 'results' / '1_feature_extraction' / '3_surface_freesurfer')
surface_feature_dir.mkdir(exist_ok=True, parents=True)
if not surface_feature_dir.exists():
    raise_error(f'Output directory {surface_feature_dir} does not exist.')
else:
    logger.info(f'Output directory set to {surface_feature_dir}.')

# input related
chunksize = 1000
categories = list(range(190, 198))
session = 'ses-2'
ukbb_cat_html = (
    "https://biobank.ctsu.ox.ac.uk/crystal/search.cgi?srch=freesurfer&yfirst="
    "2000&ylast=2022")

# %%

# UKBB showcase information
# category-atlas relation
surface_categories = ukbb_get_searched_categories_description(ukbb_cat_html)

# Parsing
start_time_fsloading_all = time.time()

with tempfile.TemporaryDirectory() as tmpdir:
    # Directories
    tmpdir = Path(tmpdir)
    install_dir = tmpdir / 'ukb_super'
    FS_fname = install_dir / FS_filename

    # Install and get UKBB FreeSurfer output .tsv from datalad super dataset
    logger.info(
        f'Create temporary directory {tmpdir} to get ukb super dataset')
    logger.info(f'Clone ukb_super datalad dataset from {REPO_URL}')
    dl.install(path=install_dir, source=REPO_URL)
    logger.info(
        f'Get FreeSurfer tsv-file {FS_fname} from ukb_super dataset in '
        f'{install_dir}.')
    dl.get(path=FS_fname, dataset=install_dir)
    logger.info(
        f'Got FreeSurfer tsv-file {FS_fname}.')

    # loop through categories (only clone dataset once)
    for category in categories:
        start_time_fsloading = time.time()
        logger.info(f'========= \n START parsing category {category}')

        # category tree
        datafields, description = ukbb_get_category_datafields(
            category, session=session)

        # output related
        surface_feature_fname = (
            surface_feature_dir /
            f'ukbb_{surface_categories[category]}_category_{category}.sqlite')
        results_uri = f'sqlite:///{surface_feature_fname.as_posix()}'

        # create iterator (FS_reader) from iterable (df)
        logger.info('Create iterator to read in tsv file.')
        FS_reader = pd.read_csv(
            FS_fname.as_posix(), sep='\t', iterator=True, chunksize=chunksize,
            usecols=datafields)

        # loop chunkwise through iterator
        for chunk in FS_reader:
            # create proper chunk-dataframe
            FS_chunk = chunk[datafields]  # resort columns in correct order
            FS_chunk.columns = description
            FS_chunk['SubjectID'] = 'sub-' + FS_chunk['SubjectID'].astype(str)
            FS_chunk.set_index(['SubjectID'], inplace=True)

            # filter NaN rows
            FS_chunk_nonan = FS_chunk[~FS_chunk.isna().any(axis=1)].copy()

            # category specific dataframe splitting and saving
            if category in [192, 195, 196, 197]:
                FS_area = (
                    FS_chunk_nonan.loc[:, FS_chunk_nonan.columns.str.startswith(
                            'Area')])
                FS_thickness = (
                    FS_chunk_nonan.loc[:, FS_chunk_nonan.columns.str.startswith(
                            'Mean')])
                FS_volume = (
                    FS_chunk_nonan.loc[:, FS_chunk_nonan.columns.str.startswith(
                            'Volume')])
                # save to SQLITE
                save_features(
                    df=FS_area,
                    uri=results_uri,
                    kind='fs_surface_area',
                    atlas_name=surface_categories[category],
                    agg_function=None
                    )
                save_features(
                    df=FS_thickness,
                    uri=results_uri,
                    kind='fs_surface_thickness',
                    atlas_name=surface_categories[category],
                    agg_function='mean'
                    )
                save_features(
                    df=FS_volume,
                    uri=results_uri,
                    kind='fs_volume',
                    atlas_name=surface_categories[category],
                    agg_function=None,
                    )
            elif category == 190:
                FS_intensity = (
                    FS_chunk_nonan.loc[:, FS_chunk_nonan.columns.str.startswith(
                            'Mean')])
                FS_volume = (
                    FS_chunk_nonan.loc[:, FS_chunk_nonan.columns.str.startswith(
                            'Volume')])
                save_features(
                    df=FS_intensity,
                    uri=results_uri,
                    kind='fs_intensity',
                    atlas_name=surface_categories[category],
                    agg_function='Mean'
                    )
                save_features(
                    df=FS_volume,
                    uri=results_uri,
                    kind='fs_volume',
                    atlas_name=surface_categories[category],
                    agg_function=None
                    )
            elif category in [191, 193, 194]:
                if category == 191:
                    kind = 'fs_volume'
                    agg_function = None
                elif category == 193:
                    kind = 'fs_surface_area'
                    agg_function = None
                elif category == 194:
                    kind = 'fs_graywhite_contrast'
                    agg_function = 'ratio'
                save_features(
                            df=FS_chunk_nonan,
                            uri=results_uri,
                            kind=kind,
                            atlas_name=surface_categories[category],
                            agg_function=agg_function
                            )

        elapsed_time_fsloading = time.time() - start_time_fsloading
        logger.info(
            'Chunkwise reading rows with filtered columns for category '
            f'{category} of Freesurfer .tsv and saving to SQLite database took '
            f'{elapsed_time_fsloading} s.')

elapsed_time_fsloading_all = time.time() - start_time_fsloading_all
logger.info(
    'All specified categories were parsed, filtered and saved and temporary '
    f'directory {tmpdir} to clone ukb super dataset was removed. This took '
    f'{elapsed_time_fsloading_all} s')
