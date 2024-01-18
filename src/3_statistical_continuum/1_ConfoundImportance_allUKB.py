# %%
# imports
import os
from pathlib import Path
from confoundcontinuum.pipelines import feature_choice
import numpy as np

import pandas as pd
import datatable as dt

from scipy.stats import pearsonr, spearmanr
from scipy.stats import pointbiserialr

from confoundcontinuum.logging import logger

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402
# %%
# set params

# input
feature = 'all_gmv'
shuffle_feature = False

# %%
# directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())

base_dir = project_dir / 'results'
feature_dir = base_dir / '1_feature_extraction' / 'extracted_features'
phenotype_dir = base_dir / '2_phenotype_extraction'

out_dir = base_dir / '3_statistical_continuum'
out_dir.mkdir(exist_ok=True, parents=True)

# fnames
# in
confound_fname = (
    phenotype_dir / '40_allUKB_reduced_cleaned_exICD10-V-VI-stroke_IMG.jay'
    )
tiv_fname = phenotype_dir / '50_TIV.csv'

# out
corr_trgt_cnfds_fname = (
    out_dir / 'summary_correlations_target_allUKB_confounds.csv')
corr_ftrs_cnfds_fname = (
    out_dir / 'summary_correlations_GMV_allUKB_confounds.csv')
p_vals_ftrs_cnfds_fname = (
    out_dir / 'summary_pvals_GMV_allUKB_confounds.csv'
)
corr_ftrs_cnfds_abs_fname = (
    out_dir / 'summary_abs_correlations_GMV_allUKB_confounds.csv')

# %%
# load data
FTR = feature_choice(feature=feature, project_dir=project_dir)
logger.info(f'Features {feature} loaded.')
CNFD = dt.fread(confound_fname)
CNFD = CNFD.to_pandas()
CNFD.set_index('SubjectID', inplace=True)
TIV = pd.read_csv(tiv_fname, index_col='SubjectID')

logger.info('Confounds (including target) and TIV loaded.')

CNFD = CNFD.join(TIV[['TIV']], how='inner')  # Add TIV to confounds
logger.info('TIV and controls merged with confounds.')

# %%
# ------------------------------------------------------------------------------
# ----------------- Define content-based groups of confounds -------------------

# Define Confound Column Groups by content
general_cols = [
    'UK_Biobank_assessment_centre-0', 'Brain_MRI_sign-off_timestamp-0_decimal',
    'Age-0', 'Sex-0',
    ]
body_measure_cols = [
    'Waist_circumference-0', 'Hip_circumference-0',
    'Comparative_body_size_at_age_10-0',  # < 10000 sbjs, discrete?
    'Comparative_height_size_at_age_10-0',  # < 10000 sbjs, discrete?
    'Handedness_chirality/laterality-0',  # < 10000 sbjs, discrete
    'Hand_grip_strength_right-0', 'Hand_grip_strength_left-0',
    'HGS_mean_left_right', 'TIV',
    'mean_Height-0', 'mean_Seated_Height-0',
    'Body_mass_index_BMI-0',
    'Weight-0', 'Body_fat_percentage-0', 'Whole_body_fat_mass-0',
    'Whole_body_fat-free_mass-0', 'Whole_body_water_mass-0',
    'Basal_metabolic_rate-0', 'Impedance_of_whole_body-0',
    'Impedance_of_leg_right-0', 'Impedance_of_leg_left-0',
    'Impedance_of_arm_right-0', 'Impedance_of_arm_left-0',
    'Leg_fat_percentage_right-0', 'Leg_fat_mass_right-0',
    'Leg_fat-free_mass_right-0', 'Leg_predicted_mass_right-0',
    'Leg_fat_percentage_left-0', 'Leg_fat_mass_left-0',
    'Leg_fat-free_mass_left-0', 'Leg_predicted_mass_left-0',
    'Arm_fat_percentage_right-0', 'Arm_fat_mass_right-0',
    'Arm_fat-free_mass_right-0', 'Arm_predicted_mass_right-0',
    'Arm_fat_percentage_left-0', 'Arm_fat_mass_left-0',
    'Arm_fat-free_mass_left-0', 'Arm_predicted_mass_left-0',
    'Trunk_fat_percentage-0', 'Trunk_fat_mass-0',
    'Trunk_fat-free_mass-0', 'Trunk_predicted_mass-0',
]
heart_cols = [  # -> no overlapping sbjs with HGS (in IMG)
    'mean_Systolic_blood_pressure-0',  # < 40000 sbjs
    'mean_Diastolic_blood_pressure-0',  # < 40000 sbjs
    'mean_Pulse_rate-0',  # < 40000 sbjs
    'Pulse_wave_Arterial_Stiffness_index-0',
    ]
bone_cols = [  # -> no overlapping sbjs with HGS (in IMG)
    'Heel_bone_mineral_density_BMD_T-score,_automated_left-0',  # < 30000
    'Heel_bone_mineral_density_BMD_T-score,_automated_right-0',  # < 30000
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_left-0',  # < 4000
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_right-0',  # < 4000
]
job_cols = [  # -> no overlapping sbjs with HGS (in IMG)
    'Time_employed_in_main_current_job-0',  # < 17000 sbjs
    'Length_of_working_week_for_main_job-0',  # < 17000 sbjs
    'Frequency_of_travelling_from_home_to_job_workplace-0',  # < 17000 sbjs
    'Distance_between_home_and_job_workplace-0',  # < 17000 sbjs
    'Job_involves_mainly_walking_or_standing-0',    # < 17000 sbjs # discrete
    'Job_involves_heavy_manual_or_physical_work-0',    # < 17000 sbjs # discrete
    'Job_involves_shift_work-0',    # < 17000 sbjs # discrete
    'Job_involves_night_shift_work-0',  # < 3000 sbjs  discrete
    'Current_employment_status-0',  # discrete
    'Transport_type_for_commuting_to_job_workplace-0',  # < 15000 sbjs
]
demogr_cols = [
    'Age_completed_full_time_education-0',  # < 10000 sbjs
    'Country_of_birth_UK/elsewhere-0',  # < 10000 sbjs, discrete
    'Breastfed_as_a_baby-0',  # < 10000 sbjs, discrete
    'Adopted_as_a_child-0',  # discrete
    'Part_of_a_multiple_birth-0',  # < 10000 sbjs
    'Maternal_smoking_around_birth-0',  # < 10000 sbjs, discrete
]
mental_health_cols = [  # discrete
    'Mood_swings-0', 'Miserableness-0', 'Irritability-0',
    'Sensitivity_/_hurt_feelings-0', 'Fed-up_feelings-0', 'Nervous_feelings-0',
    'Worrier_/_anxious_feelings-0', "Tense_/_'highly_strung'-0",
    'Worry_too_long_after_embarrassment-0', "Suffer_from_'nerves'-0",
    'Loneliness,_isolation-0', 'Guilty_feelings-0', 'Risk_taking-0',
    'Frequency_of_depressed_mood_in_last_2_weeks-0',
    'Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-0',
    'Frequency_of_tenseness_/_restlessness_in_last_2_weeks-0',
    'Frequency_of_tiredness_/_lethargy_in_last_2_weeks-0',
    'Seen_doctor_GP_for_nerves,_anxiety,_tension_or_depression-0',
    'Seen_a_psychiatrist_for_nerves,_anxiety,_tension_or_depression-0',
    'Severity_of_manic/irritable_episodes-0',
    'Length_of_longest_manic/irritable_episode-0',
    'Ever_depressed_for_a_whole_week-0',  # discrete
    'Longest_period_of_depression-0',  # < 20000 sbjs
    'Number_of_depression_episodes-0',  # < 20000 sbjs
    'Ever_unenthusiastic/disinterested_for_a_whole_week-0',  # discrete
    'Ever_manic/hyper_for_2_days-0',  # discrete
    'Ever_highly_irritable/argumentative_for_2_days-0',  # discrete
    'Longest_period_of_unenthusiasm_/_disinterest-0',  # < 12000 sbjs
    'Number_of_unenthusiastic/disinterested_episodes-0',  # < 12000 sbjs
    # 'Length_of_longest_manic/irritable_episode-0',  # < 5000 sbjs
    # 'Severity_of_manic/irritable_episodes-0',  # < 5000 sbjs
]
satisfaction_cols = [
    'Happiness-0', 'Work/job_satisfaction-0', 'Health_satisfaction-0',
    'Family_relationship_satisfaction-0', 'Friendships_satisfaction-0',
    'Financial_situation_satisfaction-0',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-0',  # discrete
]
motherhood_cols = [
    'Age_at_first_live_birth-0',  # <15000 sbjs
    'Age_at_last_live_birth-0',  # <15000 sbjs
    'Ever_had_stillbirth,_spontaneous_miscarriage_or_termination-0',  # discrete
    'Ever_taken_oral_contraceptive_pill-0',    # <22000 sbjs # discrete
    'Age_started_oral_contraceptive_pill-0',  # <22000 sbjs
    'Age_when_last_used_oral_contraceptive_pill-0',  # <22000 sbjs
    'Age_started_hormone-replacement_therapy_HRT-0',  # < 10000 sbjs
    'Age_last_used_hormone-replacement_therapy_HRT-0',  # < 10000 sbjs
    'Ever_had_hysterectomy_womb_removed-0',  # < 20000 sbjs, discrete
]
breathing_cols = [
    'mean_Forced_vital_capacity_FVC-0',  # <35000sbjs
    'mean_Forced_expiratory_volume_in_1-second_FEV1-0',  # <35000sbjs
    'mean_Peak_expiratory_flow_PEF-0',  # <35000sbjs
]
cognitive_cols = [
    'Duration_to_complete_numeric_path_trail_#1-0',  # < 32000
    'Total_errors_traversing_numeric_path_trail_#1-0',  # < 32000
    'Duration_to_complete_alphanumeric_path_trail_#2-0',  # < 32000
    'Total_errors_traversing_alphanumeric_path_trail_#2-0',  # < 32000
    'Fluid_intelligence_score-0',
    'Maximum_digits_remembered_correctly-0',  # < 33000 sbjs
    'Number_of_puzzles_correctly_solved-0',  # < 32000
    'Number_of_puzzles_correct-0',  # < 32000
    'Prospective_memory_result-0',  # discrete
    'Mean_time_to_correctly_identify_matches-0',
    'Number_of_word_pairs_correctly_associated-0',  # < 32000
    'Number_of_symbol_digit_matches_made_correctly-0',  # < 33000
]
alcohol_cols = [  # -> no overlapping sbjs with HGS (in IMG)
    'Alcohol_consumed-0',  # < 15000 sbjs
    'Red_wine_intake-0',  # < 3000 sbjs
    'Rose_wine_intake-0',  # < 500 sbjs
    'White_wine_intake-0',  # < 3000 sbjs
    'Beer/cider_intake-0',  # < 3000 sbjs
    'Fortified_wine_intake-0',  # < 200 sbjs
    'Spirits_intake-0',  # < 1000 sbjs
    'Other_alcohol_intake-0',  # < 300 sbjs
]

# %%
# ------------------------------------------------------------------------------
# --------------- Define variable-type-based groups of confounds ---------------

cont_cols = [  # use pearson correlation (or spearman as well?)
    'Hand_grip_strength_left-0', 'Hand_grip_strength_right-0',
    'HGS_mean_left_right',
    'Brain_MRI_sign-off_timestamp-0_decimal', 'TIV',
    'Waist_circumference-0', 'Hip_circumference-0',
    'mean_Height-0', 'mean_Seated_Height-0', 'Weight-0',
    'mean_Systolic_blood_pressure-0', 'mean_Diastolic_blood_pressure-0',
    'mean_Pulse_rate-0', 'Time_employed_in_main_current_job-0',
    'Length_of_working_week_for_main_job-0',
    'Frequency_of_travelling_from_home_to_job_workplace-0',
    'Distance_between_home_and_job_workplace-0',
    'Age_completed_full_time_education-0',
    'Age_at_first_live_birth-0',
    'Age_at_last_live_birth-0',
    'Age_started_oral_contraceptive_pill-0',
    'Age_when_last_used_oral_contraceptive_pill-0',
    'Age_started_hormone-replacement_therapy_HRT-0',
    'Age_last_used_hormone-replacement_therapy_HRT-0',
    'mean_Forced_vital_capacity_FVC-0',
    'mean_Forced_expiratory_volume_in_1-second_FEV1-0',
    'mean_Peak_expiratory_flow_PEF-0',
    # Attention! Values are STD from usual age normal value
    'Heel_bone_mineral_density_BMD_T-score,_automated_left-0',
    'Heel_bone_mineral_density_BMD_T-score,_automated_right-0',
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_left-0',
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_right-0',
    'Maximum_digits_remembered_correctly-0',
    'Longest_period_of_depression-0',
    'Number_of_depression_episodes-0',
    'Longest_period_of_unenthusiasm_/_disinterest-0',
    'Number_of_unenthusiastic/disinterested_episodes-0',
    'Length_of_longest_manic/irritable_episode-0',
    'Duration_to_complete_numeric_path_trail_#1-0',  # unit: deciseconds
    'Total_errors_traversing_numeric_path_trail_#1-0',
    'Duration_to_complete_alphanumeric_path_trail_#2-0',
    'Total_errors_traversing_alphanumeric_path_trail_#2-0',
    'Number_of_puzzles_correctly_solved-0',  # 0-15
    'Fluid_intelligence_score-0',
    'Mean_time_to_correctly_identify_matches-0',  # unit: ms
    'Number_of_word_pairs_correctly_associated-0',
    'Body_mass_index_BMI-0', 'Number_of_puzzles_correct-0',
    'Pulse_wave_Arterial_Stiffness_index-0', 'Body_fat_percentage-0',
    'Whole_body_fat_mass-0', 'Whole_body_fat-free_mass-0',
    'Whole_body_water_mass-0', 'Basal_metabolic_rate-0',  # kJ
    'Impedance_of_whole_body-0',  # Ohm
    'Impedance_of_leg_right-0', 'Impedance_of_leg_left-0',
    'Impedance_of_arm_right-0', 'Impedance_of_arm_left-0',
    'Leg_fat_percentage_right-0', 'Leg_fat_mass_right-0',
    'Leg_fat-free_mass_right-0', 'Leg_predicted_mass_right-0',
    'Leg_fat_percentage_left-0', 'Leg_fat_mass_left-0',
    'Leg_fat-free_mass_left-0', 'Leg_predicted_mass_left-0',
    'Arm_fat_percentage_right-0', 'Arm_fat_mass_right-0',
    'Arm_fat-free_mass_right-0', 'Arm_predicted_mass_right-0',
    'Arm_fat_percentage_left-0', 'Arm_fat_mass_left-0',
    'Arm_fat-free_mass_left-0', 'Arm_predicted_mass_left-0',
    'Trunk_fat_percentage-0', 'Trunk_fat_mass-0', 'Trunk_fat-free_mass-0',
    'Trunk_predicted_mass-0',
    'Number_of_symbol_digit_matches_made_correctly-0', 'Age-0',
]

discrete_cols = [
    'UK_Biobank_assessment_centre-0',
    'Country_of_birth_UK/elsewhere-0',
    'Handedness_chirality/laterality-0',  # 1:right, 3: both
    'Current_employment_status-0',
    'Transport_type_for_commuting_to_job_workplace-0',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-0',
]

discrete_cols_rank = [  # Use spearman rank correlation
    'Job_involves_mainly_walking_or_standing-0',
    'Job_involves_heavy_manual_or_physical_work-0',
    'Job_involves_shift_work-0', 'Comparative_body_size_at_age_10-0',
    'Comparative_height_size_at_age_10-0',
    'Frequency_of_depressed_mood_in_last_2_weeks-0',
    'Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-0',
    'Frequency_of_tenseness_/_restlessness_in_last_2_weeks-0',
    'Frequency_of_tiredness_/_lethargy_in_last_2_weeks-0',
    'Job_involves_night_shift_work-0', 'Happiness-0',
    'Work/job_satisfaction-0', 'Health_satisfaction-0',
    'Family_relationship_satisfaction-0', 'Friendships_satisfaction-0',
    'Financial_situation_satisfaction-0',  'Prospective_memory_result-0',
    'Red_wine_intake-0',  # glasses yesterday
    'Rose_wine_intake-0', 'White_wine_intake-0',
    'Beer/cider_intake-0', 'Fortified_wine_intake-0',
    'Spirits_intake-0', 'Other_alcohol_intake-0',
]

binary_cols = [  # use point biserial correlation
    'Sex-0', 'Breastfed_as_a_baby-0',  # 1: yes, 0: no
    'Adopted_as_a_child-0', 'Part_of_a_multiple_birth-0',
    'Maternal_smoking_around_birth-0',
    'Mood_swings-0', 'Miserableness-0', 'Irritability-0',
    'Sensitivity_/_hurt_feelings-0', 'Fed-up_feelings-0',
    'Nervous_feelings-0', 'Worrier_/_anxious_feelings-0',
    "Tense_/_'highly_strung'-0", 'Worry_too_long_after_embarrassment-0',
    "Suffer_from_'nerves'-0", 'Loneliness,_isolation-0', 'Guilty_feelings-0',
    'Risk_taking-0',  # 1: yes, 0: no
    'Seen_doctor_GP_for_nerves,_anxiety,_tension_or_depression-0',
    'Seen_a_psychiatrist_for_nerves,_anxiety,_tension_or_depression-0',
    'Ever_had_stillbirth,_spontaneous_miscarriage_or_termination-0',
    'Ever_taken_oral_contraceptive_pill-0',
    'Ever_had_hysterectomy_womb_removed-0',
    'Ever_depressed_for_a_whole_week-0',
    'Ever_unenthusiastic/disinterested_for_a_whole_week-0',
    'Ever_manic/hyper_for_2_days-0',
    'Ever_highly_irritable/argumentative_for_2_days-0',
    'Severity_of_manic/irritable_episodes-0', 'Alcohol_consumed-0',
]

# Columns only applying to one gender
female_cols = [
    'Ever_had_stillbirth,_spontaneous_miscarriage_or_termination-0',
    'Ever_taken_oral_contraceptive_pill-0',
    'Ever_had_hysterectomy_womb_removed-0',
    'Age_started_hormone-replacement_therapy_HRT-0',
    'Age_last_used_hormone-replacement_therapy_HRT-0',
    'Age_at_first_live_birth-0', 'Age_at_last_live_birth-0',
    'Age_started_oral_contraceptive_pill-0',
    'Age_when_last_used_oral_contraceptive_pill-0',
]

# insepct columns
# with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
#    display(CNFD.describe())

# %%
# ------------------------------------------------------------------------------
# ------------------------ TARGET CONFOUND RELATIONSHIP ------------------------
# %%
# Correlate confounds with target (HGS) (confound-target relationship)

# if correlation file already exists -> load
if os.path.isfile(corr_trgt_cnfds_fname):
    correlations = pd.read_csv(corr_trgt_cnfds_fname, index_col=[0])
    logger.info(
        f'Correlation csv was found in {corr_trgt_cnfds_fname} and loaded.')
# if correlation file does not exist -> calculate
else:
    correlations = pd.DataFrame(CNFD.columns, columns=['confounds']).copy()
    correlations.set_index('confounds', inplace=True)

    for col in CNFD.columns:
        x = CNFD.loc[:, col].values
        y = CNFD.loc[:, 'HGS_mean_left_right'].values
        NaNs = np.logical_or(np.isnan(x), np.isnan(y))
        if col in cont_cols:
            corr = pearsonr(x[~NaNs], y[~NaNs])
            corr_type = 'pearson_r'
        elif col in discrete_cols_rank:
            corr = spearmanr(x[~NaNs], y[~NaNs])
            corr_type = 'spearman_r'
        elif col in binary_cols:
            corr = pointbiserialr(x[~NaNs], y[~NaNs])
            corr_type = 'pointbiserial_r'
        elif col in discrete_cols:
            corr = (np.nan, np.nan)
            corr_type = 'None'
            logger.info(
                "Does not make sense to calculate corelation for discrete, "
                "non-rankable variables.")
        else:
            logger.warning(f'The column {col} was missed to be included.')
        correlations.loc[col, 'corr_type'] = corr_type
        correlations.loc[col, 'corr_value'] = corr[0]
        correlations.loc[col, 'p'] = corr[1]

    # assign groups to confound correlations
    for cnfd in correlations.index:
        if cnfd in general_cols:
            correlations.loc[cnfd, 'group'] = 'general'
        elif cnfd in body_measure_cols:
            correlations.loc[cnfd, 'group'] = 'body'
        elif cnfd in heart_cols:
            correlations.loc[cnfd, 'group'] = 'heart'
        elif cnfd in bone_cols:
            correlations.loc[cnfd, 'group'] = 'bone density'
        elif cnfd in job_cols:
            correlations.loc[cnfd, 'group'] = 'job'
        elif cnfd in demogr_cols:
            correlations.loc[cnfd, 'group'] = 'sociodemographics'
        elif cnfd in mental_health_cols:
            correlations.loc[cnfd, 'group'] = 'mental health'
        elif cnfd in satisfaction_cols:
            correlations.loc[cnfd, 'group'] = 'satisfaction'
        elif cnfd in motherhood_cols:
            correlations.loc[cnfd, 'group'] = 'motherhood'
        elif cnfd in breathing_cols:
            correlations.loc[cnfd, 'group'] = 'respiration'
        elif cnfd in cognitive_cols:
            correlations.loc[cnfd, 'group'] = 'cognition'
        elif cnfd in alcohol_cols:
            correlations.loc[cnfd, 'group'] = 'alcohol'

    correlations.dropna(inplace=True)  # drop NaNs of non meaningful corr cols
    correlations['corr_abs_value'] = abs(correlations['corr_value'])
    correlations.sort_values(by='corr_value', ascending=True, inplace=True)

    # Save correlation DF
    correlations.to_csv(corr_trgt_cnfds_fname)
    logger.info(
        'All correlations with target HGS were saved to '
        f'{corr_trgt_cnfds_fname}.')

# %%
# ------------------------------------------------------------------------------
# ------------------------ FEATURE CONFOUND RELATIONSHIP -----------------------
# %%
# Correlate all confounds with GMV (confound-feature relationship)

if (os.path.isfile(corr_ftrs_cnfds_fname)
        and os.path.isfile(p_vals_ftrs_cnfds_fname)):
    CORR = pd.read_csv(corr_ftrs_cnfds_fname, index_col=[0])
    p_vals = pd.read_csv(p_vals_ftrs_cnfds_fname, index_col=[0])
    logger.info(
        f'Correlation csv was found in {corr_ftrs_cnfds_fname} and loaded.')
    logger.info(
        f'p value csv was found in {p_vals_ftrs_cnfds_fname} and loaded.')
else:
    # Remove target related confounds
    trgt_cols = [
        'Hand_grip_strength_left-0', 'Hand_grip_strength_right-0',
        'HGS_mean_left_right']
    CNFD.drop(trgt_cols, axis=1, inplace=True)

    # Initialize correlation and p_value dataframe
    CORR = pd.DataFrame(index=list(FTR.columns))
    p_vals = pd.DataFrame(index=list(FTR.columns))

    for confound in CNFD.columns:
        # Get same dimensions (cnfds & features) withouth NaNs
        FTR_CNFD = FTR.join(CNFD[confound].dropna(), how='inner')

        # Correlate each confound with all features
        if confound in cont_cols:
            for feature in FTR.columns:
                CORR.loc[feature, confound] = pearsonr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[0]
                p_vals.loc[feature, confound] = pearsonr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[1]
        elif confound in discrete_cols_rank:
            for feature in FTR.columns:
                CORR.loc[feature, confound] = spearmanr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[0]
                p_vals.loc[feature, confound] = spearmanr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[1]
        elif confound in binary_cols:
            for feature in FTR.columns:
                CORR.loc[feature, confound] = pointbiserialr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[0]
                p_vals.loc[feature, confound] = pointbiserialr(
                    FTR_CNFD[confound], FTR_CNFD[feature])[1]
        elif confound in discrete_cols:
            logger.info(
                    "Does not make sense to calculate corelation for discrete, "
                    "non-rankable variables.")
        else:
            logger.warning(f'The column {col} was missed to be included.')

    # Save correlation DF
    CORR.to_csv(corr_ftrs_cnfds_fname)
    logger.info(
        f'Correlation dataframe was saved to {corr_ftrs_cnfds_fname}.')
    CORR_abs = CORR.abs()
    CORR_abs.to_csv(corr_ftrs_cnfds_abs_fname)
    logger.info(
        'Absolute correlations dataframe was saved to '
        f'{corr_ftrs_cnfds_abs_fname}.')
    # Save p_vals DF
    p_vals.to_csv(p_vals_ftrs_cnfds_fname)
    logger.info(
        f'Correlation dataframe was saved to {p_vals_ftrs_cnfds_fname}.')
