# %%
# imports
import os
from pathlib import Path
from confoundcontinuum.io import read_features
import datatable as dt

from confoundcontinuum.logging import configure_logging, log_versions, logger
configure_logging()
log_versions()

# %%
# general directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
root_dir = project_dir / 'results' / '1_feature_extraction'
surface_dir = root_dir / '3_surface_freesurfer'
fc_dir = project_dir / 'data' / 'functional'
out = root_dir / 'extracted_features'
out.mkdir(exist_ok=True, parents=True)
logger.info('Features .jay files will be saved to ' + out.as_posix())

# %%
# GMV directories and settings

feature_dirs = ['1_gmd_Schaefer', '2_gmd_SUIT', '4_gmd_tian']
feature_names = [
    '1_gmd_schaefer_all_subjects.sqlite',
    '2_gmd_SUIT_all_subjects.sqlite',
    '4_gmd_tian_all_subjects.sqlite',
    ]
atlas_names = [
    'schaefer2018_1000parcels',
    'SUITxMNI',
    'Tian4x3TxMNInonlinear2009cAsym',
    ]

# aggregation settings
win_limits = [0.1, 0.1]
agg_fct = (
        'winsorized_mean_limits_'
        + str(win_limits[0]).replace('.', '') +
        '_'+str(win_limits[1]).replace('.', '')
    )

# Read in
win_mean_df_dict = {}
for feature_dir, atlas_name, feature_name in zip(feature_dirs, atlas_names, feature_names):  # noqa
    feature_fname = root_dir / feature_dir / feature_name
    feature_uri = f'sqlite:///{feature_fname.as_posix()}'

    # Read in
    logger.info(
        f'Reading in GMV {atlas_name} from {feature_dir}')
    win_mean_df = read_features(
        uri=feature_uri,
        kind='gmd',
        atlas_name=atlas_name,
        index_col=['SubjectID', 'Session'],
        agg_function=agg_fct,
    )
    # aggregate in dict
    win_mean_df_dict[atlas_name] = win_mean_df

# keep 2nd session only
GMV = {
    key: win_mean.xs('ses-2', level=1, drop_level=True).copy()
    for (key, win_mean) in win_mean_df_dict.items()}

# convert and save
cortical_gmv_df = GMV['schaefer2018_1000parcels'].reset_index(level=0)
subcortical_gmv_df = GMV['Tian4x3TxMNInonlinear2009cAsym'].reset_index(level=0)
cerebellar_gmv_df = GMV['SUITxMNI'].reset_index(level=0)

cortical_gmv_dt = dt.Frame(cortical_gmv_df)
subcortical_gmv_dt = dt.Frame(subcortical_gmv_df)
cerebellar_gmv_dt = dt.Frame(cerebellar_gmv_df)

cortical_gmv_dt.to_jay((out / '1_gmd_schaefer_all_subjects.jay').as_posix())
subcortical_gmv_dt.to_jay((out / '4_gmd_tian_all_subjects.jay').as_posix())
cerebellar_gmv_dt.to_jay((out / '2_gmd_SUIT_all_subjects.jay').as_posix())

logger.info('Gray matter volumes converted and saved to .jay files.')

# %%
# surfaces
# load
_available_features = {
    'dk_white': 'ukbb_Freesurfer_desikan_white_category_192.sqlite',
    'dk_pial': 'ukbb_Freesurfer_desikan_pial_category_193.sqlite',
    'dk_gw': 'ukbb_Freesurfer_desikan_gw_category_194.sqlite',
    }
_feature_kinds = {
    'dk_white': ['fs_surface_area', 'fs_surface_thickness', 'fs_volume'],
    'dk_pial': ['fs_surface_area'],
    'dk_gw': ['fs_graywhite_contrast']
    }
_kind_agg = {
    'fs_surface_area': None,
    'fs_surface_thickness': 'mean',
    'fs_volume': None,
    'fs_intensity': 'Mean',
    'fs_graywhite_contrast': 'ratio'
}
feature_fnames = {
    feature_key: (surface_dir / feature_fname)
    for (feature_key, feature_fname) in _available_features.items()}
feature_furi = {
    feature_key: f'sqlite:///{feature_fname .as_posix()}'
    for (feature_key, feature_fname) in feature_fnames.items()}
atlas_names = {
    feature_key:
    (feature_fname[(feature_fname.find("ukbb_") + len("ukbb_")):
     feature_fname.find("_category")])
    for (feature_key, feature_fname) in _available_features.items()}
# load features - dictionary (data only contains 2nd session already)
fs_df = {}
for (atlas_key, atlas_name) in atlas_names.items():
    for kind in _feature_kinds[atlas_key]:
        fs_df[atlas_key+'$'+kind] = read_features(
                uri=feature_furi[atlas_key], kind=kind, atlas_name=atlas_name,
                index_col=['SubjectID'],
                agg_function=_kind_agg[kind])
logger.info('Surface features loaded.')

# remove whole-brain feature measures to make them as confounds
total_surface_cols = [
    'Area of TotalSurface (left hemisphere)',
    'Area of TotalSurface (right hemisphere)']
mean_thickness_cols = [
    'Mean thickness of GlobalMeanMean thickness (left hemisphere)',
    'Mean thickness of GlobalMeanMean thickness (right hemisphere)']
total_volume_cols = [
    'Volume of bankssts (left hemisphere)',
    'Volume of bankssts (right hemisphere)']
contrast_cols = [
    'Grey-white contrast in unknown (left hemisphere)',
    'Grey-white contrast in unknown (right hemisphere)']
total_surface_white_df = fs_df['dk_white$fs_surface_area'][total_surface_cols]
total_surface_pial_df = fs_df['dk_pial$fs_surface_area'][total_surface_cols]
mean_thickness_df = fs_df['dk_white$fs_surface_thickness'][mean_thickness_cols]
total_volume_df = fs_df['dk_white$fs_volume'][total_volume_cols]
total_contrast_df = fs_df['dk_gw$fs_graywhite_contrast'][contrast_cols]
fs_df['dk_white$fs_surface_area'].drop(total_surface_cols, axis=1, inplace=True)
fs_df['dk_pial$fs_surface_area'].drop(total_surface_cols, axis=1, inplace=True)
fs_df['dk_white$fs_surface_thickness'].drop(
    mean_thickness_cols, axis=1, inplace=True)
fs_df['dk_white$fs_volume'].drop(
    total_volume_cols, axis=1, inplace=True)
fs_df['dk_gw$fs_graywhite_contrast'].drop(
    total_contrast_df, axis=1, inplace=True)
# change column names of white and pial surface area for identifiability
fs_df['dk_white$fs_surface_area'].columns = [
    'white_' + x for x in fs_df['dk_white$fs_surface_area'].columns]
fs_df['dk_pial$fs_surface_area'].columns = [
    'pial_' + x for x in fs_df['dk_pial$fs_surface_area'].columns]

# convert and save
white_surface_df = fs_df['dk_white$fs_surface_area'].reset_index(level=0)
white_thickness_df = fs_df['dk_white$fs_surface_thickness'].reset_index(level=0)
white_volume_df = fs_df['dk_white$fs_volume'].reset_index(level=0)
pial_surface_df = fs_df['dk_pial$fs_surface_area'].reset_index(level=0)
gray_white_contrast_df = fs_df[
    'dk_gw$fs_graywhite_contrast'].reset_index(level=0)

white_surface_dt = dt.Frame(white_surface_df)
white_thickness_dt = dt.Frame(white_thickness_df)
white_volume_dt = dt.Frame(white_volume_df)
pial_surface_dt = dt.Frame(pial_surface_df)
gray_white_contrast_dt = dt.Frame(gray_white_contrast_df)

white_surface_dt.to_jay((out / 'dk_white_surface.jay').as_posix())
white_thickness_dt.to_jay((out / 'dk_white_thickness.jay').as_posix())
white_volume_dt.to_jay((out / 'dk_white_volume.jay').as_posix())
pial_surface_dt.to_jay((out / 'dk_pial_surface.jay').as_posix())
gray_white_contrast_dt.to_jay((out / 'dk_gray_white_contrast.jay').as_posix())

logger.info('Surface features converted and saved to .jay files.')

# %%
# fc
# load
feature_fnames = fc_dir / 'Schaefer400x17_nodenoise_UKB_5000_z.csv'
fc = dt.fread(feature_fnames.as_posix()).to_pandas()
fc.rename(columns={'subID': 'SubjectID'}, inplace=True)
fc.set_index('SubjectID', inplace=True)
fc.index = 'sub-' + fc.index.astype(str)

# convert and save
fc_df = fc.reset_index(level=0)
fc_dt = dt.Frame(fc_df)
fc_dt.to_jay((out / 'fc_Schaefer400x17_nodenoise_5000_z.jay').as_posix())

logger.info('FC features converted and saved to .jay files.')
