# imports
import glob
import os
from pathlib import Path
import pandas as pd

# parameters
out_name = 'summary-GMVFC_CC.csv'
out_dir_name = 'predictions_GMVFC_CC'

# directories
# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
out_dir = project_dir / 'results' / '4_predictions' / out_dir_name
out_fname = out_dir / out_name

# get all csv files in out_dir
extension = 'csv'
os.chdir(out_dir)
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

# export to .csv
combined_csv.to_csv(out_fname, index=False)
