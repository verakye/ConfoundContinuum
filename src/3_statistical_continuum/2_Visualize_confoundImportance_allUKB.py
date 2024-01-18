# %%
# imports
import os
from pathlib import Path
from matplotlib import pyplot as plt
from confoundcontinuum.logging import logger
import pandas as pd
# from scipy.stats import shapiro

import seaborn as sns
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Arial"

# %%
# paths
project_dir = Path(os.getcwd())
io_dir = project_dir / 'results' / '3_statistical_continuum'
io_dir.mkdir(exist_ok=True, parents=True)
plot_dir = io_dir / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

# fnames
# in
corr_trgt_cnfds_fname = (
    io_dir / 'summary_correlations_target_allUKB_confounds.csv')
corr_ftrs_cnfds_fname = (
    io_dir / 'summary_correlations_GMV_allUKB_confounds.csv')
p_vals_ftrs_cnfds_fname = (
    io_dir / 'summary_pvals_GMV_allUKB_confounds.csv'
)
corr_ftrs_cnfds_abs_fname = (
    io_dir / 'summary_abs_correlations_GMV_allUKB_confounds.csv')

# load data
# load correlations with target
if os.path.isfile(corr_trgt_cnfds_fname):
    correlations = pd.read_csv(corr_trgt_cnfds_fname, index_col=[0])
    logger.info(
        f'Correlations with target was loaded from {corr_trgt_cnfds_fname}.')
# drop correlations with HGS itself
correlations.drop([
    'Hand_grip_strength_right-0', 'Hand_grip_strength_left-0',
    'HGS_mean_left_right'], axis=0, inplace=True)
logger.info('Correlations with target loaded.')

# load correlations with features
if (os.path.isfile(corr_ftrs_cnfds_fname)
        and os.path.isfile(p_vals_ftrs_cnfds_fname)):
    CORR = pd.read_csv(corr_ftrs_cnfds_fname, index_col=[0])
    p_vals = pd.read_csv(p_vals_ftrs_cnfds_fname, index_col=[0])
    logger.info(
        f'Correlations with features was loaded from {corr_ftrs_cnfds_fname}.')
    logger.info(
        f'p vals corr with features was loaded from {p_vals_ftrs_cnfds_fname}.')
logger.info('Correlations with features loaded.')

# %%
# Adjust CORR dataframe for plotting

# sort columns based on mean of correlations in column
CORR_sort = CORR[CORR.mean(axis=0).sort_values(ascending=True).index].copy()

# # check for normality to see if Fisher z-transformation needed
# CORR_sort_shap = CORR_sort.apply(shapiro)
# logger.info(
#     f'n={sum(CORR_sort_shap.loc[1,:]<.05)} columns NOT normally distributed.')

# # Fisher z-transform correlations for less biased averaging
# CORR_sort_z = np.arctanh(CORR_sort)
# corr_sort_z_se = 1/np.sqrt(1088-3)  # N is no. of parcels

# melt dataframe
CORR_melt = CORR_sort.melt(var_name='Confound', value_name='Correlation')

# add groups per confound (according to cnfd-target correlations)
for cnfd in correlations.index:
    CORR_melt.loc[
        CORR_melt['Confound'] == cnfd,
        'group'] = correlations.loc[cnfd, 'group']
    CORR_melt.loc[
        CORR_melt['Confound'] == cnfd,
        '25%'] = CORR_sort.describe().loc['25%', cnfd]
    CORR_melt.loc[
        CORR_melt['Confound'] == cnfd,
        '75%'] = CORR_sort.describe().loc['75%', cnfd]

# %%
# Names plotting

confound_names_plotting = {
    'Leg_fat_percentage_left-0': 'Fat left leg (%)',
    'Leg_fat_percentage_right-0': 'Fat right leg (%)',
    'Impedance_of_arm_left-0': 'Impedance left arm',
    'Impedance_of_arm_right-0': 'Impedance right arm',
    'Impedance_of_whole_body-0': 'Impedance whole body',
    'Body_fat_percentage-0': 'Fat body (%)',
    'Arm_fat_percentage_left-0': 'Fat left arm (%)',
    'Arm_fat_percentage_right-0': 'Fat right arm (%)',
    'Leg_fat_mass_left-0': 'Fat left leg (mass)',
    'Leg_fat_mass_right-0': 'Fat right leg (mass)',
    'Impedance_of_leg_right-0': 'Impedance right leg',
    'Impedance_of_leg_left-0': 'Impedance left leg',
    'Trunk_fat_percentage-0': 'Fat trunk (%)',
    'Mean_time_to_correctly_identify_matches-0': 'Time to correctly identify matches (mean)',  # noqa
    'Age-0': 'Age',
    'Worrier_/_anxious_feelings-0': 'Worrier/anxious feelings',
    'Whole_body_fat_mass-0': 'Fat whole body (mass)',
    'Sensitivity_/_hurt_feelings-0': 'Sensitivity/hurt feelings',
    'mean_Pulse_rate-0': 'Pulse rate (mean)',
    'Arm_fat_mass_left-0': 'Fat left arm (mass)',
    'Arm_fat_mass_right-0': 'Fat right arm (mass)',
    'Age_started_oral_contraceptive_pill-0': 'Age started oral contraceptive pill',  # noqa
    'Guilty_feelings-0': 'Guilty feelings',
    'Worry_too_long_after_embarrassment-0': 'Worry too long after embarrassment',  # noqa
    'Seen_doctor_GP_for_nerves,_anxiety,_tension_or_depression-0': 'Seen doctor (GP) nerves/anxiety/tension/depression',  # noqa
    'Beer/cider_intake-0': 'Beer/cider intake',
    'Frequency_of_tiredness_/_lethargy_in_last_2_weeks-0': 'Frequency tiredness/lethargy (last 2_weeks)',  # noqa
    'Work/job_satisfaction-0': 'Job satisfaction',
    'Nervous_feelings-0': 'Nervous feelings',
    'Miserableness-0': 'Miserableness',
    'Ever_had_hysterectomy_womb_removed-0': 'Ever had hysterectomy womb removed',  # noqa
    'Ever_depressed_for_a_whole_week-0': 'Ever depressed (whole week)',
    'Ever_unenthusiastic/disinterested_for_a_whole_week-0': 'Ever unenthusiastic/disinterested (whole week)',  # noqa
    'Age_last_used_hormone-replacement_therapy_HRT-0': 'Age last used hormone-replacement therapy',  # noqa
    'Loneliness,_isolation-0': 'Loneliness/isolation',
    'Duration_to_complete_alphanumeric_path_trail_#2-0': 'TMT-B (duration)',
    'Number_of_word_pairs_correctly_associated-0': 'Number of word pairs correctly associated',  # noqa
    'Longest_period_of_depression-0': 'Longest period depression',
    'Fed-up_feelings-0': 'Fed-up feelings',
    'Duration_to_complete_numeric_path_trail_#1-0': 'TMT-A (duration)',
    'Frequency_of_depressed_mood_in_last_2_weeks-0': 'Frequency depressed mood (last 2 weeks)',  # noqa
    "Tense_/_'highly_strung'-0": "Tense/'highly strung'",
    'Total_errors_traversing_alphanumeric_path_trail_#2-0': 'TMT-B (errors)',
    'Longest_period_of_unenthusiasm_/_disinterest-0': 'Longest period unenthusiasm/disinterest',  # noqa
    'Mood_swings-0': 'Mood swings',
    'Health_satisfaction-0': 'Health satisfaction',
    'Total_errors_traversing_numeric_path_trail_#1-0': 'TMT-A (errors)',
    'Frequency_of_tenseness_/_restlessness_in_last_2_weeks-0': 'Frequency tenseness/restlessness (last 2 weeks)',  # noqa
    'Happiness-0': 'Happiness 0',
    'Prospective_memory_result-0': 'Prospective memory result ',
    'Ever_highly_irritable/argumentative_for_2_days-0': 'Ever highly irritable/argumentative (2 days)',  # noqa
    'Family_relationship_satisfaction-0': 'Family/relationship satisfaction',
    'Age_when_last_used_oral_contraceptive_pill-0': 'Age last used oral contraceptive pill',  # noqa
    'Seen_a_psychiatrist_for_nerves,_anxiety,_tension_or_depression-0': 'Seen psychiatrist nerves/anxiety/tension/depression',  # noqa
    'Financial_situation_satisfaction-0': 'Financial situation satisfaction',
    'Job_involves_mainly_walking_or_standing-0': 'Job mainly walking/standing',
    'Brain_MRI_sign-off_timestamp-0_decimal': 'Scan-time',
    'Length_of_longest_manic/irritable_episode-0': 'Length longest manic/irritable episode',  # noqa
    'Severity_of_manic/irritable_episodes-0': 'Severity manic/irritable episodes',  # noqa
    'Part_of_a_multiple_birth-0': 'Part of multiple birth',
    'Adopted_as_a_child-0': 'Adopted as child',
    'Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-0': 'Frequency unenthusiasm/disinterest (last_2_weeks)',  # noqa
    'Number_of_unenthusiastic/disinterested_episodes-0': 'Number unenthusiastic/disinterested episodes',  # noqa
    'Maternal_smoking_around_birth-0': 'Maternal smoking around birth',
    "Suffer_from_'nerves'-0": "Suffer from 'nerves'",
    'Comparative_body_size_at_age_10-0': 'Comparative body size age 10',
    'Number_of_depression_episodes-0': 'Number depression episodes',
    'Irritability-0': 'Irritability',
    'Job_involves_shift_work-0': 'Job involves shift',
    'Ever_manic/hyper_for_2_days-0': 'Ever manic/hyper (2 days)',
    'Time_employed_in_main_current_job-0': 'Time employed current job',
    'Ever_had_stillbirth,_spontaneous_miscarriage_or_termination-0': 'Ever had stillbirth/spontaneous/miscarriage/termination',  # noqa
    'Other_alcohol_intake-0': 'Other alcohol intake',
    'Age_started_hormone-replacement_therapy_HRT-0': 'Age started hormone-replacement therapy',  # noqa
    'Hip_circumference-0': 'Hip circumference',
    'Comparative_height_size_at_age_10-0': 'Comparative height age 10',
    'Rose_wine_intake-0': 'Rose wine intake',
    'Job_involves_heavy_manual_or_physical_work-0': 'Job involves heavy manual/physical work',  # noqa
    'Ever_taken_oral_contraceptive_pill-0': 'Ever taken oral contraceptive pill',  # noqa
    'Trunk_fat_mass-0': 'Fat trunk (mass)',
    'Frequency_of_travelling_from_home_to_job_workplace-0': 'Frequency commute home-job',  # noqa
    'Age_completed_full_time_education-0': 'Age completed full time education',
    'Alcohol_consumed-0': 'Alcohol consumed',
    'Red_wine_intake-0': 'Red wine intake',
    'Fluid_intelligence_score-0': 'Fluid intelligence score',
    'White_wine_intake-0': 'White wine intake',
    'Friendships_satisfaction-0': 'Friendships satisfaction',
    'Breastfed_as_a_baby-0': 'Breastfed as baby',
    'mean_Systolic_blood_pressure-0': 'Systolic blood pressure (mean)',
    'Number_of_symbol_digit_matches_made_correctly-0': 'Number symbol digit matches made correctly',  # noqa
    'Body_mass_index_BMI-0': 'Body mass index (BMI)',
    'Fortified_wine_intake-0': 'Fortified wine intake',
    'Number_of_puzzles_correct-0': 'Number of puzzles correct',
    'Maximum_digits_remembered_correctly-0': 'Maximum digits remembered correctly',  # noqa
    'Spirits_intake-0': 'Spirits intake',
    'Age_at_last_live_birth-0': 'Age last live birth',
    'Pulse_wave_Arterial_Stiffness_index-0': 'Pulse wave Arterial Stiffness index',  # noqa
    'Age_at_first_live_birth-0': 'Age first live birth',
    'Distance_between_home_and_job_workplace-0': 'Distance home-job',
    'Job_involves_night_shift_work-0': 'Job involves night shift work',
    'Number_of_puzzles_correctly_solved-0': 'Number of puzzles correctly solved',  # noqa
    'Risk_taking-0': 'Risk taking',
    'mean_Diastolic_blood_pressure-0': 'Diastolic blood pressure (mean)',
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_right-0': 'Mineral density right heel bone (T-score, manual)',  # noqa
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_left-0': 'Mineral density left heel bone (T-score, manual)',  # noqa
    'Heel_bone_mineral_density_BMD_T-score,_automated_left-0': 'Mineral density left heel bone (T-score, automated)',  # noqa
    'Length_of_working_week_for_main_job-0': 'Length working week main job',
    'Heel_bone_mineral_density_BMD_T-score,_automated_right-0': 'Mineral density right heel bone (T-score, automated)',  # noqa
    'Waist_circumference-0': 'Waist circumference',
    'Weight-0': 'Weight',
    'TIV': 'TIV',
    'mean_Peak_expiratory_flow_PEF-0': 'Peak expiratory flow (PEF, mean)',
    'mean_Seated_Height-0': 'Seated height (mean)',
    'mean_Forced_expiratory_volume_in_1-second_FEV1-0': 'Forced expiratory volume (1s) (FEV1, mean)',  # noqa
    'mean_Forced_vital_capacity_FVC-0': 'Forced vital capacity (FVC, mean)',
    'mean_Height-0': 'Height (mean)',
    'Leg_fat-free_mass_left-0': 'Fat-free left leg (mass)',
    'Leg_predicted_mass_left-0': 'Predicted mass left leg',
    'Leg_fat-free_mass_right-0': 'Fat-free right leg (mass)',
    'Leg_predicted_mass_right-0': 'Predicted mass right leg',
    'Basal_metabolic_rate-0': 'Basal metabolic rate',
    'Arm_fat-free_mass_left-0': 'Fat-free left arm (mass)',
    'Arm_predicted_mass_left-0': 'Predicted mass left arm',
    'Arm_fat-free_mass_right-0': 'Fat-free right arm (mass)',
    'Arm_predicted_mass_right-0': 'Predicted mass right arm',
    'Whole_body_fat-free_mass-0': 'Fat-free whole body (mass)',
    'Whole_body_water_mass-0': 'Water whole body (mass)',
    'Sex-0': 'Sex',
    'Trunk_fat-free_mass-0': 'Fat-free trunk (mass)',
    'Trunk_predicted_mass-0': 'Predicted mass trunk',
    # 'Hand_grip_strength_right-0': 'HGS right',
    # 'Hand_grip_strength_left-0': 'HGS left',
    # 'HGS_mean_left_right': 'HGS (mean)'
    }

# %%
# ------------------------------------------------------------------------------
# ------------------------ Visualize target & feature correlations -------------

# Plot corr with sign
correlations.sort_values(by='corr_value', ascending=True, inplace=True)

# sort in same order as target correlations
sorter = correlations.index.to_list()
CORR_melt_trgt_sort = CORR_melt.sort_values(
    by="Confound",
    key=lambda column: column.map(lambda e: sorter.index(e)),
    inplace=False)

# %%
# Visualize

sns.set_style({'font.family': 'Arial'})

# define big color palette
colors = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff', '#9A6324',
    '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
    '#ffffff', '#000000']  # removed '#bfef45', at 9th position

font_size = 18
font_size_2 = 20
fig, (ax1, ax2) = plt.subplots(figsize=(16, 35), nrows=1, ncols=2, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0)

# set grey rectangle
left, bottom, width, height = (-.3, -1, .6, 150)
rect = mpatches.Rectangle(
    (left, bottom), width, height,
    # fill=False,
    alpha=0.25,
    facecolor="grey")
ax1.add_patch(rect)
left, bottom, width, height = (-.075, -1, .15, 150)
rect = mpatches.Rectangle(
    (left, bottom), width, height,
    # fill=False,
    alpha=0.25,
    facecolor="grey")
ax2.add_patch(rect)

# plot barplots
tgrt_h = sns.barplot(
    x='corr_value', y=correlations.index, data=correlations, hue='group',
    dodge=False, ax=ax1, palette=colors, alpha=1, edgecolor='grey',
    linewidth=0.5)

# ftr_h = sns.barplot(
#     x="Correlation", y="Confound", data=CORR_melt_trgt_sort, hue='group',
#     dodge=False, ax=ax2, palette=colors, errorbar='sd', errwidth=.5,
#     capsize=.15, alpha=1, edgecolor='grey', linewidth=0.5)

ftr_h = sns.boxplot(
    data=CORR_melt_trgt_sort, x="Correlation", y="Confound", hue='group',
    dodge=False, ax=ax2, palette=colors, saturation=1, width=.75, fliersize=.5,
    linewidth=0.5, whis=2)

# adapt (axes) labels
ax1.yaxis.set_ticklabels(
    list(confound_names_plotting.values()), fontsize=font_size)
ax1.set_ylabel('Confound', fontsize=font_size_2)
ax2.set_ylabel(None)
ax1.xaxis.set_ticklabels(
    [f'{i:.1f}' for i in tgrt_h.get_xticks()], fontsize=font_size_2)
ax2.xaxis.set_ticklabels(
    [f'{i:.1f}' for i in ftr_h.get_xticks()], fontsize=font_size_2)
ax1.set_xlabel(None)
ax2.set_xlabel(None)
# ax.legend(fontsize=font_size, title_fontsize=font_size)
ax1.grid(color='grey', alpha=0.25, linestyle='-', linewidth=0.5)
ax1.axvline(x=0, ymin=-1, ymax=150, color='grey', linewidth=0.5)
ax2.grid(color='grey', alpha=0.25, linestyle='-', linewidth=0.5)
ax2.axvline(x=0, ymin=-1, ymax=150, color='grey', linewidth=0.5)
fig.text(.48, .093, 'Correlation', ha='left', fontsize=font_size_2)
handles, labels = ax2.get_legend_handles_labels()
# handles = ax2.get_legend()
fig.legend(
    handles, labels,
    loc=(0.6, 0.4), fontsize=font_size_2, title_fontsize=font_size_2,
    framealpha=1)
ax1.legend_.remove()
ax2.legend_.remove()

# save figure
barplots_fname = (
        plot_dir / 'HGS_GMV_allUKB_correlations_median.pdf')
plt.savefig(barplots_fname.as_posix(), bbox_inches='tight')
logger.info(
    'Visualization of correlations with features and target saved to '
    f'as {barplots_fname}.')

plt.show()

# %%
