# %%
import os
from pathlib import Path
import tempfile

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

# %% define variables and paths

# URL to datalad dataset to clone in temporary directory to get file names
REPO_URL = 'ria+http://ukb.ds.inm7.de#~cat_m0wp1'

# RUN THINGS IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())

# directory of actual python script to be run for feature extraction
script_dir = project_dir / 'src' / '1_feature_extraction'

# check existance results directory
results_dir = (
    project_dir / 'results' / '1_feature_extraction' / '4_gmd_tian' /
    'databases')
results_dir.mkdir(exist_ok=True, parents=True)

# check existance log directory
logs_dir = (
    project_dir / 'results' / '1_feature_extraction' / '4_gmd_tian' / 'logs')
logs_dir.mkdir(exist_ok=True, parents=True)

submit_fname = script_dir / '5_gmd_tian_dag.submit'
dag_fname = script_dir / '5_gmd_tian_dag.dag'

# %% define preamble

# define arguments for executable here
exec_string = (
    '5_gmd_Tian.py '
    '--subid $(subject) '
    '--ses $(session) '
    '--aggmethod winsorized_mean mean std '
    f'--results {results_dir.as_posix()}'
    '/5_gmd_tian_$(subject)_$(session).sqlite'
)

preamble = f"""
# The environment
universe = vanilla
getenv = True

# Resources
request_cpus = 1
request_memory = 1.6G
request_disk = 500

# Executable
initial_dir = {script_dir}
executable = $(initial_dir)/run_in_venv.sh
transfer_executable = False

arguments = {exec_string}

# Logs
log = {logs_dir}/5_gmd_tian_$(subject)_$(session).log
output = {logs_dir}/5_gmd_tian_$(subject)_$(session).out
error = {logs_dir}/5_gmd_tian_$(subject)_$(session).err
"""

with open(submit_fname, 'w') as submit_file:
    submit_file.write(preamble)
    submit_file.write('queue\n')

# %% Get subject and session name from datalad dataset and create dag-file

# Clone dataset into temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    dl.install(path=tmpdir, source=REPO_URL)  # type: ignore
    # path were symbolic links lie
    db_dir = Path(tmpdir) / 'm0wp1'

    # get all filenames which end with .nii.gz in directory in list
    files = [x.name for x in db_dir.glob('*.nii.gz')]


with open(dag_fname, 'w') as dag_file:
    # Get all subject and session names from file list
    for i_job, fname in enumerate(files):
        sub_number = fname.split('_')[0][5:]
        ses_number = fname.split('_')[1]

        dag_file.write(f'JOB job{i_job} {submit_fname}\n')
        dag_file.write(f'VARS job{i_job} subject="{sub_number}" '
                       f'session="{ses_number}"\n\n')
