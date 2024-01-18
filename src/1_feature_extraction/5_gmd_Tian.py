# %%
# import packages
from pathlib import Path
import tempfile
import time
from argparse import ArgumentParser

import pandas as pd

from confoundcontinuum.logging import configure_logging, log_versions
from confoundcontinuum.logging import logger
import confoundcontinuum.atlases as atl
from confoundcontinuum.features import get_gmd, get_vbm
from confoundcontinuum.io import save_features

# workaround to import datalad when using with ipykernel
import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

# %%
# configure logging

configure_logging()
log_versions()

# %%
# set up

# fix definitions
CAT_REPO_URL = 'ria+http://ukb.ds.inm7.de#~cat_m0wp1'
dataset_name = 'cat_m0wp1'
atlas_names = [x for x in atl.list_atlases() if x.startswith('Tian')]

# pipeline help (parser)
parser = ArgumentParser(
    description='Extract grey matter density (GMD) of VBM data per ROI. '
    'INPUT parameters required: --results, --rois, --subid, --ses. '
    'INPUT parameters optional: --tmp, --atlasdir, --winlim. '
    'See parameter help for more information.'
    ' ROIs are defined by the Tian subcortical atlas (Tian et al., 2020). '
    ' Different valid scales from I to IV can be applied. It can be chosen '
    ' if an atlas derive from a 3T or 7T scane is used. '
    ' Dimensionality within ROIs is reduced by mean of GMD per ROI,'
    ' winsorized mean (default limits 10%) and standard deviation. These'
    ' values are exported in a SQLite database to the directory specified '
    'in --results.'
)

# default inputs
# default_tmp_dir = '/tmp'
default_win_limits = [0.1, 0.1]

# PARSER INPUT ARGUMENTS

# DATA INPUT related
# subject ID
parser.add_argument(
    '--subid', metavar='subid', type=str, required=True,
    help='Subject ID in accordance with the subject IDs from the respective '
         f'database input files (see datlad dataset {CAT_REPO_URL}).')

# Session ID
parser.add_argument(
    '--ses', metavar='session', type=str, required=True,
    help='Session ID in accordance with the recording session indicated in the'
         f' VBM input files (see datalad dataset {CAT_REPO_URL}). Needed for '
         'unambigous subject-recording distinction. '
         'Valid input: "ses-2" or "ses-3".')

# DATA OUTPUT related
# aggregation methods (defaults are set in motorpred.features.get_gmd())
parser.add_argument(
    '--aggmethod', metavar='aggmethod', type=str, nargs='+',
    help='Aggregation method to summarize gray matter density per ROI.'
         'Check valid inputs via helper of motorpred.featured.get_gmd().')

# limits for winsorizing mean
parser.add_argument(
    '--winlim', metavar='winlim', type=float, nargs='+',
    help='Lower and upper limit for application of winsorized mean to '
         'aggregate GMD per ROI. The limits need to be provided as 2 '
         'floats between 0 and 1 in decimal notation of per cent values (e.g.'
         ' 0.1 0.1).')

# results directory
# path to where to store results (pipeline output)
parser.add_argument(
    '--results', metavar='results', type=str, required=True,
    help='Path where to store the results as SQLite database, '
         'containing chosen aggregation_methods of GMD per ROI. '
         'Specify as </path/to/results>/<name_of_database.sqlite>. ')

# pass input parameters to variables
args = parser.parse_args()
subid = args.subid
session = args.ses
agg_methods = args.aggmethod
win_limits = args.winlim
results_path = Path(args.results)

# check parsed arguments and give user info
# atlasname (atlas name checked in motorpred.atlases._retrieve_tian)
# aggregation methods (validity checked in motorpred.features._get_funcbyname)
# winsorize mean limits (check if argument needed,
# validity of limits checked in motorpred.features._get_funcbyname)
if ('winsorized_mean' in agg_methods) and (win_limits is None):
    logger.warning(
        "The limits argument for the aggregation option \'winsorized mean\'"
        "is required but was not specified. The default limits as defined "
        "in motorpred.features.get_gmd will therefore be used.")
elif ('winsorized_mean' not in agg_methods) and (win_limits is not None):
    logger.warning(
        "The limits for aggregation option \'winsorized mean\' were set "
        "although the \'winsorized mean\' was not chosen as aggregation "
        "option.")
# results (check existance of parent directory without DB file name!!)
results_path.parent.mkdir(exist_ok=True, parents=True)
results_uri = f'sqlite:///{results_path.as_posix()}'
logger.info('Aggregated GMD per ROI will be saved (results directory) in '
            f'{results_path.as_posix()}')

# %%
# process (only one granularity at a time processed)

start_time = time.time()

# Clone dataset into temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    # sub-directories
    tmp_data = Path(tmpdir) / 'data'
    tmp_masks = tmp_data / 'masks'  # directory to save atlases
    tmp_masks.mkdir(exist_ok=True, parents=True)

    # VBM - clone, get, load
    logger.info('Retrieve VBM nifti from datalad dataset with repo URL '
                f'{CAT_REPO_URL} to temporary directory {tmp_data}.')
    vbm_nifti = get_vbm(CAT_REPO_URL, tmp_data, dataset_name, subid, session)

    # set atlas parameters
    for atlas_name in atlas_names:
        scale = atl._available_atlases[atlas_name]['scale']
        field = atl._available_atlases[atlas_name]['magneticfield']
        space = atlas_name.split('x')[-1]
        resolution = min(
            atl._available_atlases[atlas_name]['valid_resolutions'])
        # get atlas
        atlas_img, atlas_labels, _ = atl.load_atlas(
            name=atlas_name, atlas_dir=tmp_masks, resolution=resolution,
            scale=scale, space=space, magneticfield=field)

        # resample atlas and get GMD
        logger.info(
            f'Start GMD computation for {atlas_name}.')
        gmd_parcellations, agg_func_params = get_gmd(
            atlas_img, vbm_nifti, aggregation=agg_methods, limits=win_limits)
        # Get actually used winlimits
        win_limits = agg_func_params['winsorized_mean']['limits']

        # save GMD
        for agg_name in gmd_parcellations.keys():
            # create dataframe
            logger.info(f'Create dataframe for {agg_name} for GMD.')
            gmd_df = pd.DataFrame(
                gmd_parcellations[agg_name].reshape(
                    -1, len(gmd_parcellations[agg_name])),
                index=[subid], columns=atlas_labels)
            gmd_df.index.name = 'SubjectID'
            gmd_df['Session'] = session
            gmd_df = gmd_df.reset_index().set_index(['SubjectID', 'Session'])

            # save in SQLite
            logger.info(f'Export dataframe for {agg_name} to SQLite database '
                        f'in "{results_path.as_posix()}".')
            if agg_name == 'winsorized_mean':
                save_features(
                    df=gmd_df,
                    uri=results_uri,
                    kind='gmd',
                    atlas_name=atlas_name,
                    agg_function='winsorized_mean_limits_'
                                 + str(win_limits[0]).replace('.', '') +
                                 '_'+str(win_limits[1]).replace('.', '')
                    )
            else:
                save_features(
                    df=gmd_df,
                    uri=results_uri,
                    kind='gmd',
                    atlas_name=atlas_name,
                    agg_function=agg_name
                    )
        logger.info('Dataframes exported as SQLite database.')

        # info and compute time
        elapsed_time = time.time() - start_time
        logger.info(
            'PARTIAL-PROCESSING DONE for GMD computation (including saving '
            f'results) 1 sbj, for atlas {atlas_name}. Elapsed time: '
            f'{elapsed_time} s.\n')

# info and compute time
elapsed_time = time.time() - start_time
logger.info('PROCESSING DONE for GMD computation (including saving '
            f'results) 1 sbj, for all atlas_names {atlas_names}. Elapsed total '
            f'time: {elapsed_time} s.\n')

# %%
