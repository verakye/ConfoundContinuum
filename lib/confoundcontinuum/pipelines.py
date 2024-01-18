import os
from pathlib import Path
import pandas as pd
import datatable as dt

from confoundcontinuum.logging import logger, raise_error
from confoundcontinuum.ml import heuristic_C
from confoundcontinuum._classes import HeuristicWrapper, ConfoundRemover

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


def feature_choice(feature=None, project_dir=None):
    """
    Load the different neuroimaging derived features."""

    if feature is None:
        raise_error('No feature was provided.')
    if project_dir is None:
        project_dir = Path(os.getcwd())
        logger.info(
            'No project directory was provided. Using current working directory'
            f' {project_dir}.')
    base_dir = (
        project_dir / 'results' / '1_feature_extraction' / 'extracted_features')

    if feature == 'all_gmv':
        fname_cortical = base_dir / '1_gmd_schaefer_all_subjects.jay'
        fname_subcortical = base_dir / '4_gmd_tian_all_subjects.jay'
        fname_cerebellar = base_dir / '2_gmd_SUIT_all_subjects.jay'

        cortical_dt = dt.fread(fname_cortical.as_posix())
        subcortical_dt = dt.fread(fname_subcortical.as_posix())
        cerebellar_dt = dt.fread(fname_cerebellar.as_posix())

        cortical_df = cortical_dt.to_pandas()
        subcortical_df = subcortical_dt.to_pandas()
        cerebellar_df = cerebellar_dt.to_pandas()

        cortical_df.set_index('SubjectID', inplace=True)
        subcortical_df.set_index('SubjectID', inplace=True)
        cerebellar_df.set_index('SubjectID', inplace=True)

        feature_df = pd.concat(
            [cortical_df, subcortical_df, cerebellar_df],
            axis=1, join="inner").copy()

    elif feature == 'white_thickness':
        fname = base_dir / 'dk_white_thickness.jay'
        feature_dt = dt.fread(fname.as_posix())
        feature_df = feature_dt.to_pandas()
        feature_df.set_index('SubjectID', inplace=True)

    elif feature == 'FC':
        fname = base_dir / 'fc_Schaefer400x17_nodenoise_5000_z.jay'
        feature_dt = dt.fread(fname.as_posix())
        feature_df = feature_dt.to_pandas()
        feature_df.set_index('SubjectID', inplace=True)

    return feature_df


# -----------------------------------------------------------------------------#
# Model choice for multiple algorithm/confound removal combinations
# -----------------------------------------------------------------------------#


def model_choice(pipe, confounds=None, cat_columns=None, cont_columns=None):
    """
    Define the different pipelines."""

    # preprocessing
    if cat_columns is None:
        cat_columns = []  # e.g. ['Sex']
    if cont_columns is None:
        cont_columns = []

    # z-score continous features and confounds
    preprocessor = ColumnTransformer(  # ordering problematic if cat features!
        transformers=[
            ("cont", StandardScaler(), cont_columns),  # order important!
            ("cat", "passthrough", cat_columns)],
        )  # default: remainder='drop'

    if confounds is None:
        n_cnfds = 0
    else:
        n_cnfds = len(confounds)

    # define pipelines
    if pipe == 'svr_heuristic_zscore':
        nested = False
        grid = []
        general_estimator = SVR(kernel='linear')
        pipeline = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=n_cnfds),
            HeuristicWrapper(general_estimator, heuristic_C)
            )
    elif pipe == 'linear_svr_L1_heuristic_zscore':
        nested = False
        grid = []
        general_estimator = LinearSVR(
            loss='epsilon_insensitive',
            dual=True,  # primal not supported for L1
        )
        pipeline = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=n_cnfds),
            HeuristicWrapper(general_estimator, heuristic_C)
            )
    elif pipe == 'linear_svr_L2_heuristic_zscore':
        nested = False
        grid = []
        general_estimator = LinearSVR(
            loss='squared_epsilon_insensitive',
            dual=False,
        )
        pipeline = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=n_cnfds),
            HeuristicWrapper(general_estimator, heuristic_C)
            )
    elif pipe == 'ridgeCV_zscore':
        nested = False
        grid = []
        alphas = [10, 100, 1e3, 1e4, 1e5, 1e6]
        pipeline = make_pipeline(
            preprocessor,
            # last n_confounds columns in X will be used as confounds.
            ConfoundRemover(n_confounds=n_cnfds),
            RidgeCV(
                alphas=alphas, store_cv_values=True,
                scoring="neg_root_mean_squared_error")
            )
    elif pipe == 'svr_zscore':
        nested = True
        grid = [
            {
                'svr__kernel': ['rbf'],
                'svr__C': [.05, .1, .3],
                'svr__gamma': ['scale'],
                'svr__epsilon': [.1, .5, .6],
            },
        ]
        pipeline = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=n_cnfds),
            SVR()
            )

    return pipeline, nested, grid
