# %% import packages

from pathlib import Path
import tempfile
import time
import shutil
import os

from argparse import ArgumentParser

from nilearn import image
# from nilearn import plotting
from nilearn import datasets
from nilearn import masking
# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import winsorize
import pandas as pd

from confoundcontinuum.logging import configure_logging, log_versions
from confoundcontinuum.logging import logger, raise_error
from confoundcontinuum.io import save_features
# from confoundcontinuum.io import read_features

# workaround to import datalad when using with ipykernel
import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

# %% configure logging

configure_logging()
log_versions()
# %% Parameters (INPUT)

# Fixed definitions
REPO_URL = 'ria+http://ukb.ds.inm7.de#~cat_m0wp1'
atlas_resolution = 1  # in mm; use resolution 1 to downsample to 1.5
yeo_networks = 7
_valid_nrois = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# RUN THINGS IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
default_atlas_dir = project_dir / 'data' / 'masks' / 'schaefer_2018'
default_win_limits = [0.1, 0.1]

# HELP description
parser = ArgumentParser(
    description='Extract grey matter density (GMD) of VBM data per ROI. '
    'INPUT parameters required: --results, --rois, --subid, --ses. '
    'INPUT parameters optional: --tmp, --atlasdir, --winlim. '
    'See parameter help for more information.'
    ' ROIs are defined by the Schaefer atlas (Schaefer et al., 2018). '
    ' Different granularities between 100 and 1000 (steps of 100) can be'
    ' applied. Dimensionality within ROIs is reduced by mean of GMD per ROI,'
    ' winsorized mean (default limits 10%) and standard deviation. These'
    ' values are exported in a SQLite database to the directory specified '
    'in --results.'
)

# REQUIRED Input Parameters
parser.add_argument(
    '--results', metavar='results', type=str, required=True,
    help='Path where to store the results as SQLite database, '
         'containing winsorized mean, mean and std of GMD per ROI. '
         'Specify as </path/to/results>/<name_of_database.sqlite>. ')

# atlas granularity
parser.add_argument(
    '--rois', metavar='rois', type=int, required=True, nargs='+',
    help='List of atlas granularities to be used. To be passed as integers. '
         f'Possible values are {_valid_nrois}')

# subject ID
parser.add_argument(
    '--subid', metavar='subid', type=str, required=True,
    help='Subject ID in accordance with the subject IDs from the respective '
         f'database input files (see datlad dataset {REPO_URL}).')

# Session ID
parser.add_argument(
    '--ses', metavar='session', type=str, required=True,
    help='Session ID in accordance with the recording session indicated in the'
         ' VBM input files (see datalad dataset {REPO_URL}). Needed for '
         'unambigous subject-recording distinction. '
         'Valid input: "ses-2" or "ses-3".')

# OPTIONAL Input Parameters (default values defined)

# path to store atlas downloaded via nilearn - defaults to juseless path
parser.add_argument(
    '--atlasdir', metavar='atlasdir', type=str, default=default_atlas_dir,
    help='Path where to find the atlases. Schaefer atlases are copied '
         '(when existing) from this directory to juseless local node. '
         'If not existing in the directory specified here they are '
         'downloaded using nilearn `datasets.fetch_atlas_schaefer_2018` '
         'function. '
         f'Defaults to {default_atlas_dir}')

# limits for winsorizing mean
parser.add_argument(
    '--winlim', metavar='winlim', type=float, nargs='+',
    default=default_win_limits,
    help='Lower and upper limit for application of winsorized mean to '
         'aggregate GMD per ROI. The limits need to be provided as 2 '
         'floats between 0 and 1 in decimal notation of per cent values (e.g.'
         ' 0.1 0.1).')


# pass input parameters to variables
args = parser.parse_args()

results_path = Path(args.results)
n_rois_atlas = args.rois
subid = args.subid
session = args.ses
atlas_dir = Path(args.atlasdir)
win_limits = args.winlim

# USER INFORMATION: confirm input parameters
logger.info(f'Loading input datalad dataset (VBM niftis) from URL {REPO_URL}')
logger.info('Storing results (aggregated GMD per ROI) in '
            f'{results_path.as_posix()}')
if any(a_n not in _valid_nrois for a_n in n_rois_atlas):
    raise_error(
        f'Wrong number of ROIs for the atlas: {n_rois_atlas}. '
        f'Valid options are: {_valid_nrois}')
else:
    logger.info('Using atlas granularity(ies) (no. of ROIs) for '
                f'GMD calculation of {n_rois_atlas}.')

if atlas_dir == default_atlas_dir:
    logger.info(
        f'Reading atlas from default directory: {atlas_dir.as_posix()}')
else:
    logger.info(f'Reading atlas from {atlas_dir.as_posix()} '
                f'(not default directory).')

if len(win_limits) != 2:
    raise_error('--win-limits should have exactly two elements')

if any(x < 0 or x > 1 for x in win_limits):
    raise_error('--win-limits must be between 0 and 1')

if win_limits == default_win_limits:
    logger.info(f'Using winsorized mean limits {win_limits} (default).')
else:
    logger.info(f'Using winsorized mean limits {win_limits} (not-default).')

# %% Clone, get and load VBM

start_time_all = time.time()

# Clone dataset into temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    # Define tmp subdirectories
    tmp_db_path = Path(tmpdir) / 'data'
    tmp_cat_inst_dir = tmp_db_path / 'cat_m0wp1'
    tmp_cat_dat_dir = tmp_cat_inst_dir / 'm0wp1'
    tmp_masks = tmp_db_path / 'masks'
    tmp_masks_sub = tmp_masks / 'schaefer_2018'

    # Create the directories if they do not exist
    tmp_cat_inst_dir.mkdir(exist_ok=True, parents=True)
    tmp_masks_sub.mkdir(exist_ok=True, parents=True)

    # Clone VBM datalad dataset
    logger.info(
        f'Cloning VBM datalad dataset from {REPO_URL} to {tmp_cat_inst_dir}')
    dl.install(path=tmp_cat_inst_dir, source=REPO_URL)  # type: ignore
    logger.info('Dataset cloned.')

    # VBM nifti image path (filename format UKB/VBM pre-processed specific)
    image_fname = tmp_cat_dat_dir / f'm0wp1{subid}_{session}_T1w.nii.gz'

    # check if symbolic link of cloned file exists
    if not image_fname.is_symlink():
        raise_error(f'VBM image file name "m0wp1{subid}_{session}_T1w.nii.gz" '
                    f'does not exist in: {image_fname.as_posix()}')

    # Get data (dataset component=VBM nifti) of predefined subject(s)/session(s)
    logger.info('Getting VBM niftis (dataset components) of predefined '
                'subject(s)/session(s).')
    dl.get(path=image_fname, dataset=tmp_cat_inst_dir)
    logger.info('Got VBM niftis.')

    # load VBM as nifti, resolution 1.5 mm
    logger.info(f'Loading VBM nifti from {image_fname}')
    vbm_img = image.load_img(image_fname.as_posix())
    logger.info('VBM nifti loaded.')

    # control plot
    # plotting.plot_anat(vbm_img)
    # plotting.show()

    # Get atlas

    # for loop for multiple granularities passed as ROI parameter
    for roi_atlas in n_rois_atlas:
        start_time_1gran = time.time()
        logger.info(f'Compute GMD for atlas with granularity {roi_atlas}:')

    # Get atlas
        atlas_fname = (
            atlas_dir / f'Schaefer2018_{roi_atlas}Parcels_'
            f'{yeo_networks}Networks_order_FSLMNI152_'
            f'{atlas_resolution}mm.nii.gz')
        atlas_tname = atlas_dir / f'Schaefer2018_{roi_atlas}Parcels_'\
            f'{yeo_networks}Networks_order.txt'

        logger.info(f'Checking if atlas {atlas_fname.as_posix()} and its '
                    f'metadata file {atlas_tname.as_posix()} exist and '
                    f'can be copied to {tmp_masks.as_posix()}:')

    # Fetch atlas from nilearn to /tmp
        if atlas_fname.exists() and atlas_tname.exists():
            logger.info(f'Atlas {atlas_fname.as_posix()} and its metadata '
                        'file exist.')

            logger.info(f'Copying atlas {atlas_fname.as_posix()} and '
                        f'metadata {atlas_tname.as_posix()} '
                        f'to {tmp_masks_sub.as_posix()}')
            shutil.copy2(atlas_fname, tmp_masks_sub)
            shutil.copy2(atlas_tname, tmp_masks_sub)
        else:
            logger.info(
                f'Atlas or metadata was not found in {atlas_dir.as_posix()}. '
                'They will be downloaded from nilearn.')

        atl = datasets.fetch_atlas_schaefer_2018(
            n_rois=roi_atlas, resolution_mm=atlas_resolution,
            data_dir=tmp_masks.as_posix())
        logger.info(f'Atlas now available in {tmp_masks.as_posix()}')

    # Load atlas, downsample, retrieve info
        logger.info(f'Loading atlas as nifti from {atl.maps}')
        atlas_img = image.load_img(atl.maps)  # nifti
        logger.info(f'Atlas was loaded. Granularity: {roi_atlas}, '
                    f'resolution: {atlas_resolution} mm.')

        # downsample to 1.5 mm
        logger.info('Re-sampling atlas.')
        atlas_img_re = image.resample_to_img(
            atlas_img, vbm_img, interpolation='nearest',
            )
        logger.info(
            f'Atlas was re-sampled from {atlas_img.header.get_zooms()[0]} '
            f'mm to resolution of VBM ({vbm_img.header.get_zooms()[0]} mm)'
            ' with nearest interpolation.')

        # plot check
        # plotting.plot_anat(atlas_img_re)
        # plotting.show()

        # get atlas as np array to check for granularity
        atlas = image.get_data(atlas_img_re)

        # atlas granularity/number of ROIs
        if np.unique(atlas).shape[0]-1 != roi_atlas:
            raise_error('Granularity of loaded atlas '
                        f'({np.unique(atlas).shape[0]-1}) differs from wished'
                        f' granularity ({roi_atlas}).')
        granularity = np.unique(atlas).shape[0]

        # get list of atlas labels from atlas object
        labels = [
            '_'.join(x.split('_')[1:]) for x in atl.labels.astype('U')
        ]
    # nifti - make mask & (winsorized) mean - all ROIs
        start_time_mask = time.time()

        win_mean_gmd = np.ones(shape=(granularity-1, 1)) * np.nan
        mean_gmd = np.ones(shape=(granularity-1, 1)) * np.nan  # comparison
        std_gmd = np.ones(shape=(granularity-1, 1)) * np.nan

        # extract (win.) mean of all ROIs (slow solution)
        for i_roi, roi in enumerate(range(1, granularity)):
            # apply function to image - here: select ROI
            logger.info(f'compute mask for ROI {roi}.')
            mask = image.math_img(f'img=={roi}', img=atlas_img_re)
            # apply mask to vbm data
            logger.info(f'Mask computed. Apply mask for ROI {roi}.')
            gmd = masking.apply_mask(imgs=vbm_img, mask_img=mask)
            # extract winsorized mean, mean, std of GMD within ROI
            logger.info(f'Mask applied for ROI {roi}. '
                        f'Compute winsorized mean (limits {win_limits}), mean '
                        f'and standard deviation of GMD for ROI {roi}.')
            win_mean_gmd[i_roi] = winsorize(gmd, limits=win_limits).mean()
            mean_gmd[i_roi] = gmd.mean()
            std_gmd[i_roi] = gmd.std()
            logger.info(
                f'winsorized mean, mean and STD computed for ROI {roi}.\n')

        if (
                win_mean_gmd.shape[0] != roi_atlas or
                mean_gmd.shape[0] != roi_atlas or
                std_gmd.shape[0] != roi_atlas
        ):
            raise_error('Number of ROIs computed does not'
                        ' match atlas granularity.')

        # time estimate
        elapsed_time_mask = time.time() - start_time_mask
        logger.info('Elapsed time for 1sbj mask application: '
                    f'{elapsed_time_mask} s.')

        # plot_roi on top of vbm - control
        # plotting.plot_roi(mask, vbm_img)
        # plotting.show()

    # create dataframes

        # winsorized mean of ROI GMD
        logger.info('Creating dataframes for winsorized mean, mean and STD '
                    'for each ROI GMD with multiindex subject ID and session'
                    ' ID.')
        gmd_win_df = pd.DataFrame(
            win_mean_gmd.T, index=[subid], columns=labels,
            )
        gmd_win_df.index.name = 'SubjectID'
        gmd_win_df['Session'] = session
        # multi-indexing for unambigue recording identification
        gmd_win_df = gmd_win_df.reset_index().set_index(
            ['SubjectID', 'Session'])

        # mean of ROI GMD
        gmd_mean_df = pd.DataFrame(
            mean_gmd.T, index=[subid], columns=labels,
            )
        gmd_mean_df.index.name = 'SubjectID'
        gmd_mean_df['Session'] = session
        gmd_mean_df = gmd_mean_df.reset_index().set_index(
            ['SubjectID', 'Session'])

        # STD of ROI GMDs
        gmd_std_df = pd.DataFrame(std_gmd.T, index=[subid], columns=labels)
        gmd_std_df.index.name = 'SubjectID'
        gmd_std_df['Session'] = session
        gmd_std_df = gmd_std_df.reset_index().set_index(
            ['SubjectID', 'Session'])
        logger.info('Dataframes created.\n')

    # export dataframes using SQLite

        # make results directory (without DB file name!!) if does not yet exist
        results_dir = results_path.parent
        results_dir.mkdir(exist_ok=True, parents=True)

        # Best: use directly juseless project folder as results_path
        # to save sqlite database
        results_uri = f'sqlite:///{results_path.as_posix()}'
        atlas_name = f'schaefer2018_{roi_atlas}parcels'

        logger.info('Exporting dataframes as SQLite database '
                    f'to "{results_path.as_posix()}".')

        save_features(
            df=gmd_win_df,
            uri=results_uri,
            kind='gmd',
            atlas_name=atlas_name,
            agg_function=(
                'winsorized_mean_limits_'
                + str(win_limits[0]).replace('.', '') +
                '_'+str(win_limits[1]).replace('.', '')),
                    )
        save_features(
            df=gmd_mean_df,
            uri=results_uri,
            kind='gmd',
            atlas_name=atlas_name,
            agg_function='mean',
                    )
        save_features(
            df=gmd_std_df,
            uri=results_uri,
            kind='gmd',
            atlas_name=atlas_name,
            agg_function='std',
                    )
        logger.info('Dataframes exported as SQLite database.')

        logger.info(f'COMPUTATION of GMD with 1 GRANULARITY {roi_atlas} DONE.')
        elapsed_time_1gran = time.time() - start_time_1gran
        logger.info(
            'Elapsed time for GMD computation (including saving results) '
            '1 sbj, 1 Schaefer atlas granularity'
            f' {roi_atlas}: {elapsed_time_1gran} s.')

elapsed_time_all = time.time() - start_time_all
logger.info('\n PROCESSING DONE for GMD with all passed granularities '
            f'({n_rois_atlas}). Entire process took {elapsed_time_all} s.\n')

# %% Test read_features from SQLite

# win_mean_df = read_features(uri=results_uri,
#                             kind='gmd',
#                             atlas_name=atlas_name,
#                             index_col=['SubjectID', 'Session'],
#                             agg_function='winsorized_mean_limits_'
#                             f'{win_limits[0]}_{win_limits[1]}'
#                             )

# mean_df = read_features(uri=results_uri,
#                         kind='gmd',
#                         atlas_name=atlas_name,
#                         index_col=['SubjectID', 'Session'],
#                         agg_function='mean'
#                         )

# std_df = read_features(uri=results_uri,
#                        kind='gmd',
#                        atlas_name=atlas_name,
#                        index_col=['SubjectID', 'Session'],
#                        agg_function='std'
#                        )

# %%
