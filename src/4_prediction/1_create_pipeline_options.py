# imports
import itertools
import os
from pathlib import Path
from confoundcontinuum.logging import configure_logging
from confoundcontinuum.logging import logger
configure_logging()

# parameters
out_dir_name = 'predictions_GMVCTFC_CC'

# directories
# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
out_dir = project_dir / 'results' / '4_predictions' / out_dir_name
out_dir.mkdir(exist_ok=True, parents=True)
# create logs directory for ht_condor
log_dir = out_dir / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)

# output name
job_options_fname = out_dir / 'job_options_GMVCTFC_CC.txt'

# target options
target_options = [
    "HGS_mean_left_right",
]

# brain feature options
brain_feature_options = [
    "all_gmv",
    "FC",
]

# confound feature options
confound_feature_options = [
    "None",
]

# pipeline options
pipeline_options = [
    "ridgeCV_zscore",
    "svr_zscore",
]

# confound options
confound_options = [
    "None",
    "Sex§Age",
    "Sex", "Age",
]

# combine
a = [
    target_options, brain_feature_options, confound_feature_options,
    pipeline_options, confound_options]
combinations = list(itertools.product(*a))

# drop brain-cnfd_ftr both None and cnfd-cnfd_ftr both non-None combinations
combinations = [
    tup for tup in combinations if not (tup[1] == 'None' and tup[2] == 'None')]
combinations = [
    tup for tup in combinations if not (tup[2] != 'None' and tup[4] != 'None')]

# write to file 1 combination/line (to be read by .submit)
with open(job_options_fname, 'w') as job_file:
    job_file.write(
        '\n'.join(
            f'{tup[0]} {tup[1]} {tup[2]} {tup[3]} {tup[4]}'
            for tup in combinations))

logger.info(
    'All combinations of features, pipelines and confounds were written to '
    f'{job_options_fname}.')
