# %%
# import packages and configurations

from ast import literal_eval
import math
import sys
import os
from pathlib import Path
import timeit
import joblib

import pandas as pd
import datatable as dt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from confoundcontinuum.pipelines import feature_choice, model_choice
from confoundcontinuum.visualize import visualize_predictions
from confoundcontinuum.ml import pearson_scorer, spearman_scorer

from sklearn.model_selection import GridSearchCV, KFold, \
    RepeatedStratifiedKFold, RepeatedKFold, train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from confoundcontinuum.logging import configure_logging
from confoundcontinuum.logging import log_versions
from confoundcontinuum.logging import logger

# Configurations
configure_logging()
log_versions()

# %%
# set params

# input (defined in 1_create_pipeline_options.py and passed via .submit)
target_name = sys.argv[1]
brain_feature = sys.argv[2]
confound_feature = sys.argv[3].split('§')
pipe = sys.argv[4]
cnfds = sys.argv[5].split('§')
out_dir_name = sys.argv[6]  # e.g. 'predictions_GMVFC_CC'

# None inputs
# brain_features
if brain_feature == 'None':
    brain_feature = None
    brain_feature_name = 'NoBrainFtrs'
else:
    brain_feature_name = brain_feature
logger.info(f'brain feature(s): {brain_feature}')
# confound_features
if confound_feature == ['None']:
    confound_feature = None
    confound_feature_name = 'NoCnfdFtrs'
else:
    confound_feature_name = 'CnfdFtrs_' + '_'.join(confound_feature)
logger.info(f'confound feature(s): {confound_feature}')
# confounds
if cnfds == ['None']:
    cnfds = None
    cnfds_name = 'None'
else:
    cnfds_name = '_'.join(cnfds)
logger.info(f'confounds: {cnfds}')

# Categorical columns
cat_columns = [  # general possibly ones, will be ignored if not in cnfds or cf
    'Sex',
    'UK_Biobank_assessment_centre-0',
]
if cnfds is None and confound_feature is None:
    cat_cols = None
elif cnfds is not None and confound_feature is not None:
    cat_cols = [
        elem for elem in cat_columns if elem in cnfds or confound_feature]
elif cnfds is not None and confound_feature is None:
    cat_cols = [
        elem for elem in cat_columns if elem in cnfds]
elif cnfds is None and confound_feature is not None:
    cat_cols = [
        elem for elem in cat_columns if elem in confound_feature]
logger.info(f'categorical columns: {cat_cols}')

# fixed
random_state = 43
bin_number_age_stratification = 2
bin_number_target_stratification = 2

strat_vars = ['Sex', 'AgeBinned', 'TargetBinned']
strat_groups_outer = 'StratEncod'
test_split_size = 0.2
lock_split_size = 0.1  # keep 10% of data untouched for final predictions

# nested, grid
_, nested, grid = model_choice(pipe)  # define pipeline below to have cont_cols

# cv
k_inner = 5 if nested else None
k_outer = 5
n_outer = 1
scoring_outer = {  # note: even though sex used as feature, HGS still continous
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
    "pearson_r": pearson_scorer,
    "spearman_r": spearman_scorer}
scoring_inner = {
    "RMSE": "neg_root_mean_squared_error"}
refit_inner = "RMSE"

# save names
summaryDF_save_name = (
    "summary-" + brain_feature_name + '-' + confound_feature_name + '-' +
    target_name + '-' + pipe + '-' + cnfds_name + '.csv')

# %%
# paths
# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())
root_dir = project_dir / 'results'

# input
feature_dir = root_dir / '1_feature_extraction' / 'extracted_features'
phenotype_dir = root_dir / '2_phenotype_extraction'

# output
out_dir = root_dir / '4_predictions' / out_dir_name
out_dir.mkdir(exist_ok=True, parents=True)
out_dir_sub = out_dir / (brain_feature_name + '-' + target_name + '-' + pipe)
out_dir_sub.mkdir(exist_ok=True, parents=True)
plot_dir = out_dir_sub / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

# fnames
target_fname = phenotype_dir / '20_HGS_exICD10-V-VI-stroke_IMG_noNaN-noOL.csv'
confound_fname = (
    phenotype_dir / '40_allUKB_reduced_cleaned_exICD10-V-VI-stroke_IMG.jay'
    )
tiv_fname = phenotype_dir / '50_TIV.csv'
summary_df_fname = out_dir / summaryDF_save_name

# load data
if brain_feature is not None:
    FTR = feature_choice(feature=brain_feature)
TRGT = pd.read_csv(target_fname, index_col=['SubjectID'])
CNFD = dt.fread(confound_fname)
CNFD = CNFD.to_pandas()
CNFD.set_index('SubjectID', inplace=True)
CNFD.rename(columns={"Age-0": "Age", "Sex-0": "Sex"}, inplace=True)
TIV = pd.read_csv(tiv_fname, index_col='SubjectID')
CNFD = CNFD.join(TIV[['TIV']], how='inner')  # Add TIV to confounds

# connect features
if brain_feature is None and confound_feature is not None:
    FTR = CNFD[confound_feature]
elif brain_feature is not None and confound_feature is not None:
    FTR = FTR.join(CNFD[confound_feature], how='inner')

# strat variables
STRAT = TRGT[[target_name, 'Age', 'Sex']].copy()

logger.info('Features, target and confounds/controls were loaded.')

# %%
# Make data arrays

# feature (+ potentially confounds)
if cnfds is None:
    X = FTR.copy()
else:
    X = FTR.join(CNFD[cnfds], how='inner')  # last n_cnfds columns in X -> cnfds

# intersecting subjects
idx_inter = X.index.intersection(TRGT.index)
X = X.loc[idx_inter].copy()  # features + confounds
y = TRGT.loc[idx_inter, [target_name]].copy()  # target column
STRAT = STRAT.loc[idx_inter].copy()  # stratification DF

# %%
# define the pipeline

# continous columns for preprocessor
if cat_cols is not None:
    cont_cols = [col for col in X.columns.to_list() if col not in cat_cols]
else:
    cont_cols = X.columns.to_list()

# pipeline
pipeline, nested, grid = model_choice(
    pipe, confounds=cnfds, cat_columns=cat_cols, cont_columns=cont_cols)

# %%
# add discretized age and target to STRAT

if confound_feature is None:  # only stratify with brain features
    # Age (equidistant bins)
    bins = pd.cut(
        STRAT['Age'].to_list(), bins=bin_number_age_stratification,
        precision=1)  # bins.categories (intervals) and bins.codes (classes)
    STRAT['AgeBinned'] = bins.codes

    # Target (equidistant bins)
    bins = pd.cut(
        STRAT[target_name].to_list(), bins=bin_number_target_stratification,
        precision=1)
    STRAT['TargetBinned'] = bins.codes

# %%
# stratification / stratified split

if confound_feature is None:  # stratify when brain features
    # Encode to be stratified variables
    STRAT['StratEncod'] = 0
    for var_idx, var in enumerate(strat_vars):
        STRAT['StratEncod'] += (var_idx+1)*STRAT[var]

    # Stratified split (lock data) (base on STRAT b/c only intersecting sbjs)
    idx_unlock, idx_lock = train_test_split(
        np.array(STRAT.index), test_size=lock_split_size,
        random_state=random_state, shuffle=True,
        stratify=STRAT[strat_vars],
        )
    # keep unlocked subjects, save locked indices in summaryDF below
    STRAT_unlock = STRAT.loc[idx_unlock, :]

    # Stratified split (for test set) (base on STRAT b/c only intersecting sbjs)
    idx_train, idx_test = train_test_split(
        np.array(STRAT_unlock.index), test_size=test_split_size,
        random_state=random_state, shuffle=True,
        stratify=STRAT_unlock[strat_vars],
        )  # here strat based on 3 cols, in CV based on encoded col
else:
    # Non-stratified split (lock data)
    idx_unlock, idx_lock = train_test_split(
        np.array(STRAT.index), test_size=lock_split_size,
        random_state=random_state, shuffle=True,
        stratify=None,
        )
    # keep unlocked subjects, save locked indices in summaryDF below
    STRAT_unlock = STRAT.loc[idx_unlock, :]

    # Non-stratified split (for test set) (if cnfd features involved)
    idx_train, idx_test = train_test_split(
        np.array(STRAT_unlock.index), test_size=test_split_size,
        random_state=random_state, shuffle=True,
        stratify=None,
        )
    strat_vars = ['None']

# train-test split for OOS prediction
X_train = X.loc[idx_train, :]
X_test = X.loc[idx_test, :]
y_train = y.loc[idx_train, :]
y_true = y.loc[idx_test, :]  # use for out-of-sample prediction comparison
STRAT_train = STRAT_unlock.loc[idx_train, :]
STRAT_test = STRAT_unlock.loc[idx_test, :]

# %%
# load/initialize summary DF
if os.path.isfile(summary_df_fname):
    # load
    summaryDF = pd.read_csv(
        summary_df_fname, index_col=[0],
        converters={'lock_indices': literal_eval}  # load as list not as string
        )
    row_idx = summaryDF.shape[0]  # current row
    logger.info('Summary scoring dataframe exists and was loaded.')

    # add row with set params
    summaryDF.loc[row_idx, 'brain_feature'] = brain_feature_name
    summaryDF.loc[row_idx, 'confound_feature'] = confound_feature_name
    summaryDF.loc[row_idx, 'target'] = target_name
    summaryDF.loc[row_idx, 'stratification'] = [{
        'stratification_variables': [strat_vars],
        'bins_age_stratification': bin_number_age_stratification,
        'bins_target_stratification': bin_number_target_stratification,
    }]
    summaryDF.at[row_idx, 'lock_indices'] = list(idx_lock)
    summaryDF.loc[row_idx, 'pipeline'] = [{
            'pipe': pipe,
            'confounds': cnfds,
            'k_outer': k_outer,
            'n_outer': n_outer,
        }]
    summaryDF.loc[row_idx, 't_train_s'] = ''
    summaryDF.loc[row_idx, 'scoring_cv_outer'] = [{
            'MAE_mean_CV_train': '',
            'MAE_mean_CV_test': '',
            'RMSE_mean_CV_train': '',
            'RMSE_mean_CV_test': '',
            'R2_mean_CV_train': '',
            'R2_mean_CV_test': '',
            'pearsonr_mean_CV_train': '',
            'pearsonr_mean_CV_test': '',
            'spearmanr_mean_CV_train': '',
            'spearmanr_mean_CV_test': '',
        }]
    summaryDF.loc[row_idx, 'best_parameters_cv_inner'] = '',
    summaryDF.loc[row_idx, 'scoring_final'] = [{
            'MAE_test': '',
            'RMSE_test': '',
            'R2_test': '',
            'pearsonr_test': '',
            'spearmanr_test': '',
        }]
# initialize if does not yet exist as file
else:
    summaryDF = {
        'brain_feature': brain_feature_name,
        'confound_feature': confound_feature_name,
        'target': target_name,
        'total_N': '',
        'train_N': '',
        'test_N': '',
        'stratification': [{
            'stratification_variables': strat_vars,
            'bins_age_stratification': bin_number_age_stratification,
            'bins_target_stratification': bin_number_target_stratification,
        }],
        'lock_indices': [list(idx_lock)],
        'pipeline': [{
            'pipe': pipe,
            'confounds': cnfds,
            'k_outer': k_outer,
            'n_outer': n_outer,
        }],
        't_train_s': '',
        'scoring_cv_outer': [{
            'MAE_mean_CV_train': '',
            'MAE_mean_CV_test': '',
            'RMSE_mean_CV_train': '',
            'RMSE_mean_CV_test': '',
            'R2_mean_CV_train': '',
            'R2_mean_CV_test': '',
            'pearsonr_mean_CV_train': '',
            'pearsonr_mean_CV_test': '',
            'spearmanr_mean_CV_train': '',
            'spearmanr_mean_CV_test': '',
        }],
        'best_parameters_cv_inner': '',
        'scoring_final': [{
            'MAE_test': '',
            'RMSE_test': '',
            'R2_test': '',
            'pearsonr_test': '',
            'spearmanr_test': '',
        }],
        }
    row_idx = 0
    summaryDF = pd.DataFrame(data=summaryDF)
    logger.info(
        'Summary scoring dataframe for structural models was initialized.')
summaryDF.loc[row_idx, 'total_N'] = X.shape[0]
summaryDF.loc[row_idx, 'train_N'] = X_train.shape[0]
summaryDF.loc[row_idx, 'test_N'] = X_test.shape[0]

# %%
# pipeline

# outer cv strategy
if confound_feature is None:  # stratify when brain features
    outer_cv = RepeatedStratifiedKFold(  # always stratifies y
        n_splits=k_outer, n_repeats=n_outer, random_state=random_state)
    # create CV generator
    outer_cv_generator = outer_cv.split(
        X=X_train, y=STRAT_train[strat_groups_outer],
        )
else:
    outer_cv = RepeatedKFold(  # don't stratify
        n_splits=k_outer, n_repeats=n_outer, random_state=random_state)
    # create CV generator
    outer_cv_generator = outer_cv.split(
        X=X_train, y=y_train,
        )

# Train the model
if not nested:
    logger.info(f'Non nested pipeline {pipe} will be fitted.')
    starttime = timeit.default_timer()

    # check generalizability
    scores_cv = cross_validate(
        estimator=pipeline, X=X_train, y=np.ravel(y_train),
        scoring=scoring_outer, cv=outer_cv_generator,
        return_train_score=True, return_estimator=True, verbose=3, n_jobs=1,
        )
    # fit final estimator
    estimator_final = pipeline.fit(X_train, np.ravel(y_train))

    comp_time = timeit.default_timer() - starttime
    logger.debug(f'Time needed for model fitting: {comp_time}')

if nested:
    logger.info(f'Nested pipeline {pipe} will be fitted.')
    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
            estimator=pipeline, param_grid=grid,
            scoring=scoring_inner, refit=refit_inner,
            cv=inner_cv, return_train_score=True, verbose=4, n_jobs=1,
        )
    starttime = timeit.default_timer()

    # check generalizability
    scores_cv = cross_validate(
        estimator=grid_search, X=X_train, y=np.ravel(y_train),
        scoring=scoring_outer, cv=outer_cv_generator,
        return_train_score=True, return_estimator=True, verbose=3, n_jobs=1,
        )
    # fit final estimator
    estimator_final = grid_search.fit(X_train, np.ravel(y_train))
    # Annotation: this re-does the grid_search on the entire X_train, another
    # option would be to based on some criteria select the best hyperparameters
    # either across the outer folds from the inner refit best_estimator or
    # across all cv_results across the inner and outer cv

    comp_time = timeit.default_timer() - starttime
    logger.debug(f'Time needed for model fitting: {comp_time}')

    # grid search best estimators
    scores_cv_df = pd.DataFrame(scores_cv)
    best_params = {}
    for i, est in enumerate(scores_cv_df.loc[:, 'estimator']):
        best_params[f'outer_fold_{i}'] = est.best_params_
        logger.info(
            f'Best estimator for outer fold {i} is:\n {est.best_estimator_} \n'
            f'with best parameters:\n {est.best_params_} \n')
    best_params['final_estimator'] = estimator_final.best_params_
    logger.info(
        f'Best parameters of final estimator are:\n {est.best_params_} \n')

summaryDF.loc[row_idx, 't_train_s'] = comp_time
logger.info(
    f'Mean CV test of R2 (features: {brain_feature}, {confound_feature}, '
    f'pipeline: {pipe}, confounds: {cnfds}) over outer '
    f"folds and repetitions: {format(scores_cv['test_R2'].mean(), '.2f')}\n")

# %%
# store scores from CV in summary DF

summaryDF.loc[row_idx, 'scoring_cv_outer']['MAE_mean_CV_train'] = format(
    scores_cv['train_MAE'].mean() * -1, '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['MAE_mean_CV_test'] = format(
    scores_cv['test_MAE'].mean() * -1, '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['RMSE_mean_CV_train'] = format(
    scores_cv['train_RMSE'].mean() * -1, '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['RMSE_mean_CV_test'] = format(
    scores_cv['test_RMSE'].mean() * -1, '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['R2_mean_CV_train'] = format(
    scores_cv['train_R2'].mean(), '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['R2_mean_CV_test'] = format(
    scores_cv['test_R2'].mean(), '.2f')
summaryDF.loc[row_idx, 'scoring_cv_outer']['pearsonr_mean_CV_train'] = (
    format(scores_cv['train_pearson_r'].mean(), '.2f'))
summaryDF.loc[row_idx, 'scoring_cv_outer']['pearsonr_mean_CV_test'] = (
    format(scores_cv['test_pearson_r'].mean(), '.2f'))
summaryDF.loc[row_idx, 'scoring_cv_outer']['spearmanr_mean_CV_train'] = (
    format(scores_cv['train_spearman_r'].mean(), '.2f'))
summaryDF.loc[row_idx, 'scoring_cv_outer']['spearmanr_mean_CV_test'] = (
    format(scores_cv['test_spearman_r'].mean(), '.2f'))

if not nested:
    summaryDF.loc[row_idx, 'best_parameters_cv_inner'] = 'no nested CV'
else:
    summaryDF.loc[row_idx, 'best_parameters_cv_inner'] = [best_params]

    # first_outer=scores_cv_df.loc[:, 'estimator'][0]
    # first_outer_GridCV_results_df = pd.DataFrame(first_outer.cv_results_)
    # sklearn's searcher decision how to get from cv_results to best_estimator
    # first_outer_GridCV_best_estimator = first_outer.best_estimator_

# %%
# OOS prediction

if not nested:
    # uses final estimator
    y_pred = pipeline.predict(X_test)  # cnfd columns already added

elif nested:
    # uses final estimator
    y_pred = grid_search.predict(X_test)  # cnfd columns already added

# Compare prediction scores with true target
mae = format(mean_absolute_error(y_true, y_pred), '.2f')
rmse = format(math.sqrt(mean_squared_error(y_true, y_pred)), '.2f')
r2 = format(r2_score(y_true, y_pred), '.2f')
pearson_r, _ = pearsonr(y_pred, np.ravel(y_true))
spearman_r, _ = spearmanr(y_pred, np.ravel(y_true))

# store scores from CV in summary DF
summaryDF.loc[row_idx, 'scoring_final']['MAE_test'] = mae
summaryDF.loc[row_idx, 'scoring_final']['RMSE_test'] = rmse
summaryDF.loc[row_idx, 'scoring_final']['R2_test'] = r2
summaryDF.loc[row_idx, 'scoring_final']['pearsonr_test'] = format(
    pearson_r, '.2f')
summaryDF.loc[row_idx, 'scoring_final']['spearmanr_test'] = format(
    spearman_r, '.2f')

logger.info(
    f'Prediction done. With features {brain_feature}{confound_feature}, '
    f'confounds {cnfds} and pipeline {pipe} '
    f'validation-R2 is {r2}')

# %%
# save ...

# ... scoring summary DF
summaryDF.to_csv(summary_df_fname)
logger.info(
    f'Scoring summary dataframe was saved to {summary_df_fname}.')

# ... model and scores
# scores_cv dict as .csv
scores_cv_name = (
    'scores_cv-' + brain_feature_name + '-' + confound_feature_name + '-' +
    target_name + '-' + pipe + '-' + cnfds_name + '.csv')
scores_cv_fname = out_dir_sub / scores_cv_name
pd.DataFrame(scores_cv).to_csv(scores_cv_fname)
logger.info(
    f'Outer CV scores were saved to {scores_cv_fname}.')

# cv estimators from scores_cv dict as estimator
for i, estimator_cv in enumerate(scores_cv['estimator']):
    estimator_cv_name = (
        f'estimator_cv_fold{i}-' + brain_feature_name + '-' +
        confound_feature_name + '-' + target_name + '-' + pipe + '-' +
        cnfds_name)
    estimator_cv_fname = out_dir_sub / estimator_cv_name
    joblib.dump(estimator_cv, estimator_cv_fname.as_posix())
    logger.info(
        f'Estimators of all outer CV folds were saved to {estimator_cv_fname}.')

# final estimator as estimator
estimator_final_name = (
    'estimator_final-' + brain_feature_name + '-' + confound_feature_name + '-'
    + target_name + '-' + pipe + '-' + cnfds_name)
estimator_final_fname = out_dir_sub / estimator_final_name
joblib.dump(estimator_final, estimator_final_fname.as_posix())
logger.info(
    f'Final estimator on X_train was saved to {estimator_final_fname}.')

# ... y_true, y_predicted from final estimator for plots
targets = pd.DataFrame(y_pred, columns=['y_pred'], index=y_true.index).copy()
targets['y_true'] = y_true
prediction_name = (
    'predictions-' + brain_feature_name + '-' + confound_feature_name + '-' +
    target_name + '-' + pipe + '-' + cnfds_name + '.csv')
prediction_fname = out_dir_sub / prediction_name
targets.to_csv(prediction_fname)
logger.info(
    'True and predicted target from final estimator were saved to '
    f'{prediction_fname}.')

# %%
# visualization

colors = ["#023047", "#0077B6", "#8ECAE6", "#E9C46A", "#F4A261", "#E76F51"]

for color in colors:
    fig_name = (
        f'plot{color}-' + brain_feature_name + '-' + confound_feature_name +
        '-' + target_name + '-' + pipe + '-' + cnfds_name + '.pdf')
    fig_fname = plot_dir / fig_name
    error_measures = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'pearsonr': format(pearson_r, '.2f'),
        }
    visualize_predictions(
        y_true=targets['y_true'], y_pred=targets['y_pred'],
        error_measures=error_measures, fig_fname=fig_fname,
        color=color, set_axes_labels=False, font_size=30)
    logger.info(
        'visualization of true versus predicted target was saved '
        f'to {fig_fname}')
