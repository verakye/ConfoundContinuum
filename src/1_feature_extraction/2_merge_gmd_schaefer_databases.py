# %%
# load packages
import os
from pathlib import Path
from argparse import ArgumentParser
from confoundcontinuum.logging import configure_logging, log_versions
from confoundcontinuum.logging import logger
import pandas as pd
from confoundcontinuum.io import read_features, save_features

# %% configure logging

configure_logging()
log_versions()

# %%
# Fixed definitions
_valid_nrois = list(range(100, 1100, 100))

# Default input parameters
# RUN THINGS IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
root_dir = (
    project_dir / 'results' / '1_feature_extraction' / '1_gmd_Schaefer' /
    'databases')
default_out_db_name = '1_gmd_schaefer_all_subjects.sqlite'
default_win_limits = [0.1, 0.1]
default_agg_funcs = ['win_mean', 'mean', 'std']

# %%
# INPUT Parameters

# Help description
parser = ArgumentParser(
    description='Merges single subject gray matter density (GMD) databases '
    '(with all computed parcellations) into one database for all subjects. '
    'Single subject GMD databases are the output of the script '
    '1_gmd_schaefer.py. '
    'INPUT parameters required: --rois '
    'INPUT parameters optional: --input, --output, --dbname, --winlim. '
    'See parameter help for more information.'
)

# REQUIRED Input parameters
# atlas granularities of databases
parser.add_argument(
    '--rois', metavar='rois', type=int, required=True, nargs='+',
    help='List of atlas granularities for which GMD was computed '
         '(i.e. which are saved in the single subject databases). '
         f'To be passed as integers. Possible values are {_valid_nrois}')

# OPTIONAL input parameters
# directory of single subject data
parser.add_argument(
    '--input', metavar='input', type=str, default=root_dir,
    help='Path where to get the single subject SQLite databases, '
         'results of the script `1_gmd_schaefer.py`. Databases contain '
         'for each subject the winsorized mean, mean and std of GMD per ROI '
         'and multiple granularities. Attention: The script takes all files '
         'from this directory!'
         'Specify as </path/to/databases> WITHOUT the name of the single '
         'subject databases.'
         'Info: This directory defaults to the juseless project '
         f'subdirectory {root_dir.as_posix()}.')

# directory to save output database
parser.add_argument(
    '--output', metavar='output', type=str, default=root_dir.parent,
    help='Path where to save the ALL subject SQLite database, '
         'merged from the single subject databases. The output databases '
         'will contain winsorized mean, mean and std of GMD per ROI '
         'at multiple granularities for ALL subjects.'
         'Specify as </path/to/save/database> WITHOUT the name of the database.'
         'Info: This directory defaults to the juseless project subdirectory '
         f'({root_dir.parent.as_posix()}.')

# name of output database
parser.add_argument(
    '--dbname', metavar='dbname', type=str, default=default_out_db_name,
    help='Namne of output database where all subject GMDs are saved. '
         'Specifiy as <name_of_db.sqlite>.'
         f'Defaults to {default_out_db_name}.'
)

# aggregation function (winsorized mean, mean, std)
parser.add_argument(
    '--aggfunc', metavar='aggfunc', type=str, nargs='+',
    default=default_agg_funcs,
    help='Aggregation functions with which the GMD per ROI was aggregated. '
         'The default value is the list of winsorized mean, mean and standard '
         'deviation: [\'win_mean\', \'mean\', \'std\']. Provide the input as '
         'strings with these abbreviations after each other.'
)

# limits for winsorizing mean
parser.add_argument(
    '--winlim', metavar='winlim', type=float, nargs='+',
    default=default_win_limits,
    help='Lower and upper limit with which the winsorized mean of '
         'GMD per ROI was computed to get the single subject databases. '
         'The limits need to be provided as 2 floats between 0 and 1 in '
         'decimal notation of per cent values (e.g. 0.1 0.1).')

# Pass input parameters to variables
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
n_rois_atlas = args.rois
out_db_name = args.dbname
agg_func = args.aggfunc
win_limits = args.winlim

# %%
# General Definitions

# list of all subject databases directories and respective SQLite URIs
in_paths = [item for item in input_dir.iterdir() if item.is_file()]
in_uris = [f'sqlite:///{in_path.as_posix()}' for in_path in in_paths]

# list of atlas names according to inout ROIs
atlas_names = [f'schaefer2018_{roi_atlas}parcels' for roi_atlas in n_rois_atlas]

# Directory of outout database for all subjects and respective URI
out_path = output_dir / out_db_name
out_uri = f'sqlite:///{out_path.as_posix()}'

# %%
# read, concatenate, write databases of all subjects for all parcellations

for atlas_name in atlas_names:
    # read all single subject databases (all entries of) input directory into
    # dataframe list
    logger.info(f'Reading all single subject databases for ROI {atlas_name} '
                f'from {input_dir.as_posix()}')
    # read, concat, write win_mean, mean and std databases
    # separately to save memory
    for agg_function in agg_func:
        # winsorized mean list
        if agg_function == 'win_mean':
            logger.info('Reading all single subject databases with '
                        f'aggregation function {agg_function}')
            df_list = [read_features(uri=uri,
                                     kind='gmd',
                                     atlas_name=atlas_name,
                                     index_col=['SubjectID', 'Session'],
                                     agg_function='winsorized_mean_limits_'
                                     + str(win_limits[0]).replace('.', '') +
                                     '_'+str(win_limits[1]).replace('.', '')
                                     )
                       for uri in in_uris]
        # mean and std list
        else:
            logger.info('Reading all single subject databases with '
                        f'aggregation function {agg_function}')
            df_list = [read_features(uri=uri,
                                     kind='gmd',
                                     atlas_name=atlas_name,
                                     index_col=['SubjectID', 'Session'],
                                     agg_function=agg_function
                                     )
                       for uri in in_uris]

        # concatenate dataframes of all subjects with certain parcellation
        logger.info(f'Concatenating single subject databases for atlas'
                    f'{atlas_name} and aggregation function {agg_function}.')
        df_agg_function = pd.concat(df_list)

        # write 1 dataframe for all subjects with certain parcellation to
        # output database
        logger.info('Writing concatenated database for all subjects and atlas '
                    f'{atlas_name} for aggregation function {agg_function} '
                    f'to {out_path.as_posix()}')
        # write winsorized mean
        if agg_function == 'win_mean':
            logger.info(f'Writing {agg_function}')
            save_features(df=df_agg_function,
                          uri=out_uri,
                          kind='gmd',
                          atlas_name=atlas_name,
                          agg_function='winsorized_mean_limits_'
                                       + str(win_limits[0]).replace('.', '') +
                                       '_'+str(win_limits[1]).replace('.', '')
                          )
        # write mean and std
        else:
            logger.info(f'Writing {agg_function}')
            save_features(df=df_agg_function,
                          uri=out_uri,
                          kind='gmd',
                          atlas_name=atlas_name,
                          agg_function=agg_function
                          )
logger.info(f'Single subject databases concatenated to {out_db_name} in '
            f'{output_dir} for ROIs {n_rois_atlas}')
logger.info('MERGE COMPUTATION COMPLETED')
