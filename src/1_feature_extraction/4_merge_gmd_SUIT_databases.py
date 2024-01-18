# %%
# imports

from pathlib import Path
from argparse import ArgumentParser
import os
import gc
import psutil

from confoundcontinuum.io import read_features, save_features
from confoundcontinuum.logging import logger, configure_logging, log_versions

# %%
# configure logging
configure_logging()
log_versions()
# %%
# default definitions

# directories
# RUN THINGS IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
root_dir = (
    project_dir / 'results' / '1_feature_extraction' / '2_gmd_SUIT' /
    'databases')
db_out_name = '2_gmd_SUIT_all_subjects_test.sqlite'

# features extraction related
atlas_names = ['SUITxMNI']
agg_functions = ['win_mean', 'mean', 'std']
win_limits = [0.1, 0.1]

# %%
# Argparsing

parser = ArgumentParser(
    description='Merges single subject gray matter density (GMD) databases '
    'into one database for all subjects. Single subject GMD databases are the '
    'output of the script 3_gmd_SUIT.py. '
    'INPUT parameters optional: --input, --output, --dbname, --winlim. '
    'See parameter help for more information.'
)

# Directories (all optional)
# directory of single subject databases
parser.add_argument(
    '--input', metavar='input', type=str, default=root_dir,
    help='Path to single subject SQLite databases, resulting from '
         '`3_gmd_SUIT.py`. Attention: The script takes all files '
         'stored in this directory!'
         'Specify as </path/to/databases> WITHOUT the name of the single '
         'subject database.')
# directory for merged output database
parser.add_argument(
    '--output', metavar='output', type=str, default=root_dir.parent,
    help='Path to save the merged ALL subject SQLite database, '
         'merged from the single subject databases. Specify as '
         '</path/to/save/database> WITHOUT the name of the database.')
# name of output database
parser.add_argument(
    '--dbname', metavar='dbname', type=str, default=db_out_name,
    help='Namne of output database in which all subject GMDs are saved. '
         'Specifiy as <name_of_db.sqlite>.')

# Content related parameters (all optional)
# aggregation function (winsorized mean, mean, std)
parser.add_argument(
    '--atlasnames', metavar='atlasnames', type=str, nargs='+',
    default=atlas_names,
    help='Name of atlas with which was used for parcellation. '
         'The default value is a list containing the SUIT atlas (Diedrichsen '
         'et al 2009) in MNI space: [\'SUITxMNI\'].')
parser.add_argument(
    '--aggfunc', metavar='aggfunc', type=str, nargs='+', default=agg_functions,
    help='Aggregation functions with which the GMD per ROI was aggregated. '
         'The default value is a list containing winsorized mean, mean and '
         'standard deviation: [\'win_mean\', \'mean\', \'std\']. Provide the '
         'input as strings with these abbreviations after each other.')
# limits for winsorizing mean
parser.add_argument(
    '--winlim', metavar='winlim', type=float, nargs='+',
    default=win_limits,
    help='Lower and upper limit with which the winsorized mean of '
         'GMD per ROI was computed to get the single subject databases. '
         'The limits need to be provided as 2 floats between 0 and 1 in '
         'decimal notation of per cent values (e.g. 0.1 0.1).')

args = parser.parse_args()
root_dir = Path(args.input)
output_dir = Path(args.output)
db_out_name = args.dbname
atlas_names = args.atlasnames
agg_functions = args.aggfunc
win_limits = args.winlim

# %%
# database related definitions

db_in_fnames = [
    db_in_fname for db_in_fname in root_dir.iterdir() if db_in_fname.is_file()]
db_in_URIs = [
    f'sqlite:///{db_in_fname.as_posix()}' for db_in_fname in db_in_fnames]
db_out_fname = output_dir / db_out_name
db_out_URI = f'sqlite:///{db_out_fname.as_posix()}'

# %%
# merge single subject databases (with memory freeing)

# measure memory
process_start = psutil.Process(os.getpid())
memory_process_start = process_start.memory_info().rss*10e-6  # in bytes
logger.debug(
    f'Memory usage before starting database loading: {memory_process_start} MB')

for atlas_name in atlas_names:
    for agg_function in agg_functions:
        for uri in db_in_URIs:
            # read single subject database
            logger.info(
                f'Read in single subject database for atlas {atlas_name} and '
                f'aggregation function {agg_function} from {uri}')
            if agg_function == 'win_mean':
                feature_df = read_features(
                    uri=uri, kind='gmd', atlas_name=atlas_name,
                    index_col=['SubjectID', 'Session'],
                    agg_function='winsorized_mean_limits_'
                    + str(win_limits[0]).replace('.', '') +
                    '_'+str(win_limits[1]).replace('.', ''))
            else:
                feature_df = read_features(
                    uri=uri, kind='gmd', atlas_name=atlas_name,
                    index_col=['SubjectID', 'Session'],
                    agg_function=agg_function)

            # write to "all_subject" database (automatically appends)
            logger.info(
                f'Write single subject database for atlas {atlas_name} '
                f'and aggregation function {agg_function} to '
                f'{db_out_URI}')
            if agg_function == 'win_mean':
                save_features(
                    df=feature_df, uri=db_out_URI, kind='gmd',
                    atlas_name=atlas_name,
                    agg_function='winsorized_mean_limits_'
                    + str(win_limits[0]).replace('.', '') +
                    '_'+str(win_limits[1]).replace('.', ''))
            else:
                save_features(
                    df=feature_df, uri=db_out_URI, kind='gmd',
                    atlas_name=atlas_name, agg_function=agg_function)

            # free memory
            del feature_df  # delete object
            gc.collect()  # garbage collection

        # measure memory
        process_end = psutil.Process(os.getpid())
        memory_process_end = process_end.memory_info().rss*10e-6
        difference_memory = memory_process_end - memory_process_start
        logger.debug(
            'Memory usage after having finished database merging loop: '
            f'{memory_process_end} MB.')
        logger.debug(
            'Memory needed for entire database merging: '
            f'{difference_memory} MB.')

logger.info(f'Single subject databases concatenated to {db_out_URI} in '
            f'{output_dir}.')
logger.info('MERGE COMPUTATION COMPLETED')
