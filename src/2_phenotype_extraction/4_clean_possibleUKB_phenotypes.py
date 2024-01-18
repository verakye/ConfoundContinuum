# %%
# imports
import os
from pathlib import Path
import numpy as np

import pandas as pd
import datatable as dt
from scipy.stats import zscore

from confoundcontinuum.logging import logger
from confoundcontinuum import targets

# %%
# directories

# RUN IN ROOT DIRECTORY OF PROJECT!
project_dir = Path(os.getcwd())

# input
phenotype_dir = project_dir / 'results' / '2_phenotype_extraction'
IMG_fname = (
    phenotype_dir / '31_allUKB_exICD10-V-VI-stroke_imgSbjs.csv')

# %%
# general definitions

# cleaned output -> see fnames in respective section

# cleaning
session2keep = 'ses-2'

# outlier removal
std_limit = 4

# %%
# 1) Prepare IMG subjects' phenotypes

# Load IMG subjects
healthy_img = pd.read_csv(
    IMG_fname.as_posix(), low_memory=False)  # b/c mixed dtypes
logger.info('Loaded the healthy imaging dataset.')

# Select session
healthy_img = targets.shape_phenotypes_ukbb(
    healthy_img, session2keep=session2keep, keeprun=True)
logger.info(f'Chose {session2keep =}. SubjectID was set as index.')

# Remove duplicated columns
dupl_cols = list(healthy_img.loc[:, healthy_img.columns.duplicated()].columns)
healthy_img = healthy_img.loc[:, ~healthy_img.columns.duplicated()]
logger.info(f"Found duplicated columns {dupl_cols}. Removed one of them.")

# %%
# 2) Drop unnecessary columns
drop_col = [
    'Weight_method-0', 'Hand_grip_dynamometer_device_ID-0',
    'Height_measure_device_ID-0', 'Manual_scales_device_ID-0',
    'Seating_box_device_ID-0', 'Tape_measure_device_ID-0',
    'Number_of_columns_displayed_in_round-1',
    'Number_of_columns_displayed_in_round-2',
    'Number_of_columns_displayed_in_round-3',
    'Number_of_rows_displayed_in_round-1',
    'Number_of_rows_displayed_in_round-2',
    'Number_of_correct_matches_in_round-1',
    'Number_of_correct_matches_in_round-2',
    'Number_of_correct_matches_in_round-3',
    'Number_of_incorrect_matches_in_round-1',
    'Number_of_incorrect_matches_in_round-2',
    'Number_of_incorrect_matches_in_round-3', 'Time_to_complete_round-1',
    'Time_to_complete_round-2', 'Time_to_complete_round-3',
    'Index_for_card_A_in_round-0', 'Index_for_card_A_in_round-1',
    'Index_for_card_A_in_round-2', 'Index_for_card_A_in_round-3',
    'Index_for_card_A_in_round-4', 'Index_for_card_A_in_round-5',
    'Index_for_card_A_in_round-6', 'Index_for_card_A_in_round-7',
    'Index_for_card_A_in_round-8', 'Index_for_card_A_in_round-9',
    'Index_for_card_A_in_round-10', 'Index_for_card_A_in_round-11',
    'Index_for_card_B_in_round-0', 'Index_for_card_B_in_round-1',
    'Index_for_card_B_in_round-2', 'Index_for_card_B_in_round-3',
    'Index_for_card_B_in_round-4', 'Index_for_card_B_in_round-5',
    'Index_for_card_B_in_round-6', 'Index_for_card_B_in_round-7',
    'Index_for_card_B_in_round-8', 'Index_for_card_B_in_round-9',
    'Index_for_card_B_in_round-10', 'Index_for_card_B_in_round-11',
    'Number_of_times_snap-button_pressed-0',
    'Number_of_times_snap-button_pressed-1',
    'Number_of_times_snap-button_pressed-2',
    'Number_of_times_snap-button_pressed-3',
    'Number_of_times_snap-button_pressed-4',
    'Number_of_times_snap-button_pressed-5',
    'Number_of_times_snap-button_pressed-6',
    'Number_of_times_snap-button_pressed-7',
    'Number_of_times_snap-button_pressed-8',
    'Number_of_times_snap-button_pressed-9',
    'Number_of_times_snap-button_pressed-10',
    'Number_of_times_snap-button_pressed-11',
    'Duration_to_first_press_of_snap-button_in_each_round-0',
    'Duration_to_first_press_of_snap-button_in_each_round-1',
    'Duration_to_first_press_of_snap-button_in_each_round-2',
    'Duration_to_first_press_of_snap-button_in_each_round-3',
    'Duration_to_first_press_of_snap-button_in_each_round-4',
    'Duration_to_first_press_of_snap-button_in_each_round-5',
    'Duration_to_first_press_of_snap-button_in_each_round-6',
    'Duration_to_first_press_of_snap-button_in_each_round-7',
    'Duration_to_first_press_of_snap-button_in_each_round-8',
    'Duration_to_first_press_of_snap-button_in_each_round-9',
    'Duration_to_first_press_of_snap-button_in_each_round-10',
    'Duration_to_first_press_of_snap-button_in_each_round-11',
    'Seating_box_height-0', 'Method_of_measuring_blood_pressure-0',
    'Method_of_measuring_blood_pressure-1',
    'Number_of_digits_to_be_memorised/recalled-0',
    'Number_of_digits_to_be_memorised/recalled-1',
    'Number_of_digits_to_be_memorised/recalled-2',
    'Number_of_digits_to_be_memorised/recalled-3',
    'Number_of_digits_to_be_memorised/recalled-4',
    'Number_of_digits_to_be_memorised/recalled-5',
    'Number_of_digits_to_be_memorised/recalled-6',
    'Number_of_digits_to_be_memorised/recalled-7',
    'Number_of_digits_to_be_memorised/recalled-8',
    'Number_of_digits_to_be_memorised/recalled-9',
    'Number_of_digits_to_be_memorised/recalled-10',
    'Number_of_digits_to_be_memorised/recalled-11',
    'Number_of_digits_to_be_memorised/recalled-12',
    'Number_of_digits_to_be_memorised/recalled-13',
    'Number_of_digits_to_be_memorised/recalled-14',
    'Number_of_digits_to_be_memorised/recalled-15',
    'Target_number_to_be_memorised-0', 'Target_number_to_be_memorised-1',
    'Target_number_to_be_memorised-2', 'Target_number_to_be_memorised-3',
    'Target_number_to_be_memorised-4', 'Target_number_to_be_memorised-5',
    'Target_number_to_be_memorised-6', 'Target_number_to_be_memorised-7',
    'Target_number_to_be_memorised-8', 'Target_number_to_be_memorised-9',
    'Target_number_to_be_memorised-10', 'Target_number_to_be_memorised-11',
    'Target_number_to_be_memorised-12', 'Target_number_to_be_memorised-13',
    'Target_number_to_be_memorised-14', 'Target_number_to_be_memorised-15',
    'Target_number_to_be_entered-0', 'Target_number_to_be_entered-1',
    'Target_number_to_be_entered-2', 'Target_number_to_be_entered-3',
    'Target_number_to_be_entered-4', 'Target_number_to_be_entered-5',
    'Target_number_to_be_entered-6', 'Target_number_to_be_entered-7',
    'Target_number_to_be_entered-8', 'Target_number_to_be_entered-9',
    'Target_number_to_be_entered-10', 'Target_number_to_be_entered-11',
    'Target_number_to_be_entered-12', 'Target_number_to_be_entered-13',
    'Target_number_to_be_entered-14', 'Target_number_to_be_entered-15',
    'Time_number_displayed_for-0', 'Time_number_displayed_for-1',
    'Time_number_displayed_for-2', 'Time_number_displayed_for-3',
    'Time_number_displayed_for-4', 'Time_number_displayed_for-5',
    'Time_number_displayed_for-6', 'Time_number_displayed_for-7',
    'Time_number_displayed_for-8', 'Time_number_displayed_for-9',
    'Time_number_displayed_for-10', 'Time_number_displayed_for-11',
    'Time_number_displayed_for-12', 'Time_number_displayed_for-13',
    'Time_number_displayed_for-14', 'Time_number_displayed_for-15',
    'Time_first_key_touched-0', 'Time_first_key_touched-1',
    'Time_first_key_touched-2', 'Time_first_key_touched-3',
    'Time_first_key_touched-4', 'Time_first_key_touched-5',
    'Time_first_key_touched-6', 'Time_first_key_touched-7',
    'Time_first_key_touched-8', 'Time_first_key_touched-9',
    'Time_first_key_touched-10', 'Time_first_key_touched-11',
    'Time_first_key_touched-12', 'Time_first_key_touched-13',
    'Time_first_key_touched-14', 'Time_first_key_touched-15',
    'Time_last_key_touched-0', 'Time_last_key_touched-1',
    'Time_last_key_touched-2', 'Time_last_key_touched-3',
    'Time_last_key_touched-4', 'Time_last_key_touched-5',
    'Time_last_key_touched-6', 'Time_last_key_touched-7',
    'Time_last_key_touched-8', 'Time_last_key_touched-9',
    'Time_last_key_touched-10', 'Time_last_key_touched-11',
    'Time_last_key_touched-12', 'Time_last_key_touched-13',
    'Time_last_key_touched-14', 'Time_last_key_touched-15',
    'Time_elapsed-0', 'Time_elapsed-1', 'Time_elapsed-2', 'Time_elapsed-3',
    'Time_elapsed-4', 'Time_elapsed-5', 'Time_elapsed-6', 'Time_elapsed-7',
    'Time_elapsed-8', 'Time_elapsed-9', 'Time_elapsed-10', 'Time_elapsed-11',
    'Time_elapsed-12', 'Time_elapsed-13', 'Time_elapsed-14', 'Time_elapsed-15',
    'Keystroke_history-0', 'Keystroke_history-1', 'Keystroke_history-2',
    'Keystroke_history-3', 'Keystroke_history-4', 'Keystroke_history-5',
    'Keystroke_history-6', 'Keystroke_history-7', 'Keystroke_history-8',
    'Keystroke_history-9', 'Keystroke_history-10', 'Keystroke_history-11',
    'Keystroke_history-12', 'Keystroke_history-13', 'Keystroke_history-14',
    'Keystroke_history-15', 'Number_entered_by_participant-0',
    'Number_entered_by_participant-1', 'Number_entered_by_participant-2',
    'Number_entered_by_participant-3', 'Number_entered_by_participant-4',
    'Number_entered_by_participant-5', 'Number_entered_by_participant-6',
    'Number_entered_by_participant-7', 'Number_entered_by_participant-8',
    'Number_entered_by_participant-9', 'Number_entered_by_participant-10',
    'Number_entered_by_participant-11', 'Number_entered_by_participant-12',
    'Number_entered_by_participant-13', 'Number_entered_by_participant-14',
    'Number_entered_by_participant-15', 'Round_of_numeric_memory_test-0',
    'Round_of_numeric_memory_test-1', 'Round_of_numeric_memory_test-2',
    'Round_of_numeric_memory_test-3', 'Round_of_numeric_memory_test-4',
    'Round_of_numeric_memory_test-5', 'Round_of_numeric_memory_test-6',
    'Round_of_numeric_memory_test-7', 'Round_of_numeric_memory_test-8',
    'Round_of_numeric_memory_test-9', 'Round_of_numeric_memory_test-10',
    'Round_of_numeric_memory_test-11', 'Round_of_numeric_memory_test-12',
    'Round_of_numeric_memory_test-13', 'Round_of_numeric_memory_test-14',
    'Round_of_numeric_memory_test-15',
    'Completion_status_of_numeric_memory_test-0',
    'Number_of_rounds_of_numeric_memory_test_performed-0',
    'Time_to_complete_test-0', 'Time_when_initial_screen_shown-0',
    'Test_completion_status-0', 'Time_to_answer-0', 'Time_screen_exited-0',
    'Duration_screen_displayed-0', 'Number_of_attempts-0',
    'PM:_initial_answer-0', 'PM:_final_answer-0', 'Final_attempt_correct-0',
    'History_of_attempts-0', 'Value_entered-0', 'Value_entered-1',
    'Value_entered-2', 'Value_entered-3', 'Value_entered-4', 'Value_entered-5',
    'Value_entered-6', 'Value_entered-7', 'Value_entered-8', 'Value_entered-9',
    'Value_entered-10', 'Value_entered-11', 'Value_entered-12',
    'Value_entered-13', 'Value_entered-14', 'Value_entered-15',
    'Value_entered-16', 'Value_entered-17', 'Item_selected_for_each_puzzle-0',
    'Item_selected_for_each_puzzle-1', 'Item_selected_for_each_puzzle-2',
    'Item_selected_for_each_puzzle-3', 'Item_selected_for_each_puzzle-4',
    'Item_selected_for_each_puzzle-5', 'Item_selected_for_each_puzzle-6',
    'Item_selected_for_each_puzzle-7', 'Item_selected_for_each_puzzle-8',
    'Item_selected_for_each_puzzle-9', 'Item_selected_for_each_puzzle-10',
    'Item_selected_for_each_puzzle-11', 'Item_selected_for_each_puzzle-12',
    'Item_selected_for_each_puzzle-13', 'Item_selected_for_each_puzzle-14',
    'Duration_spent_answering_each_puzzle-0',
    'Duration_spent_answering_each_puzzle-1',
    'Duration_spent_answering_each_puzzle-2',
    'Duration_spent_answering_each_puzzle-3',
    'Duration_spent_answering_each_puzzle-4',
    'Duration_spent_answering_each_puzzle-5',
    'Duration_spent_answering_each_puzzle-6',
    'Duration_spent_answering_each_puzzle-7',
    'Duration_spent_answering_each_puzzle-8',
    'Duration_spent_answering_each_puzzle-9',
    'Duration_spent_answering_each_puzzle-10',
    'Duration_spent_answering_each_puzzle-11',
    'Duration_spent_answering_each_puzzle-12',
    'Duration_spent_answering_each_puzzle-13',
    'Duration_spent_answering_each_puzzle-14', 'Screen_layout-1',
    'Screen_layout-2', 'Screen_layout-3', 'First_code_array_presented-0',
    'Number_of_puzzles_viewed-0', 'Number_of_puzzles_attempted-0',
    'Word_associated_with_"huge"-0', 'Word_associated_with_"happy"-0',
    'Word_associated_with_"tattered"-0', 'Word_associated_with_"old"-0',
    'Word_associated_with_"long"-0', 'Word_associated_with_"red"-0',
    'Word_associated_with_"sulking"-0', 'Word_associated_with_"pretty"-0',
    'Word_associated_with_"tiny"-0', 'Word_associated_with_"new"-0',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-0',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-1',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-2',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-3',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-4',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-5',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-6',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-7',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-8',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-9',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-10',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-11',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-12',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-13',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-14',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-15',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-16',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-17',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-18',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-19',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-20',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-21',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-22',
    'Errors_before_selecting_correct_item_in_numeric_path_(trail_#1)-23',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-0',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-1',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-2',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-3',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-4',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-5',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-6',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-7',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-8',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-9',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-10',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-11',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-12',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-13',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-14',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-15',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-16',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-17',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-18',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-19',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-20',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-21',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-22',
    'Errors_before_selecting_correct_item_in_alphanumeric_path_(trail_#2)-23',
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-0',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-1',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-2',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-3',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-4',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-5',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-6',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-7',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-8',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-9',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-10',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-11',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-12',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-13',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-14',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-15',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-16',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-17',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-18',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-19',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-20',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-21',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-22',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-23',  # noqa
    'Interval_between_previous_point_and_current_one_in_numeric_path_(trail_#1)-24',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-0',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-1',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-2',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-3',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-4',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-5',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-6',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-7',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-8',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-9',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-10',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-11',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-12',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-13',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-14',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-15',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-16',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-17',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-18',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-19',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-20',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-21',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-22',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-23',  # noqa
    'Interval_between_previous_point_and_current_one_in_alphanumeric_path_(trail_#2)-24',  # noqa
    'Weight_(pre-imaging)-0', 'When_diet_questionnaire_completion_requested-0',
    'Day-of-week_questionnaire_completion_requested-0',
    'Day-of-week_questionnaire_completed-0',
    'Hour-of-day_questionnaire_completed-0',
    'Duration_of_questionnaire-0',
    'Delay_between_questionnaire_request_and_completion-0',
    'Types_of_spread_used_on_bread/crackers-0',
    'Types_of_spread_used_on_bread/crackers-1',
    'Types_of_spread_used_on_bread/crackers-2',
    'Types_of_spread_used_on_bread/crackers-3',
    'Types_of_spread_used_on_bread/crackers-4',
    'Types_of_spread_used_on_bread/crackers-5',
    'Types_of_spread_used_on_bread/crackers-6',
    'Types_of_spread_used_on_bread/crackers-7',
    'Types_of_spread_used_on_bread/crackers-8',
    'Types_of_spread_used_on_bread/crackers-9',
    'Types_of_spread_used_on_bread/crackers-10',
    'Types_of_spread_used_on_bread/crackers-11',
    'Types_of_spread_used_on_bread/crackers-12',
    'Types_of_spreads/sauces_consumed-0', 'Types_of_spreads/sauces_consumed-1',
    'Types_of_spreads/sauces_consumed-2', 'Types_of_spreads/sauces_consumed-3',
    'Types_of_spreads/sauces_consumed-4', 'Types_of_spreads/sauces_consumed-5',
    'Types_of_spreads/sauces_consumed-6', 'Types_of_spreads/sauces_consumed-7',
    'Types_of_spreads/sauces_consumed-8', 'Types_of_spreads/sauces_consumed-9',
    'Types_of_spreads/sauces_consumed-10',
    'Types_of_spreads/sauces_consumed-11',
    'Types_of_spreads/sauces_consumed-12',
    'Types_of_spreads/sauces_consumed-13',
    'Types_of_spreads/sauces_consumed-14',
    'Types_of_spreads/sauces_consumed-15',
    'Types_of_spreads/sauces_consumed-16',
    'Type_of_meals_eaten-0', 'Type_of_meals_eaten-1', 'Type_of_meals_eaten-2',
    'Type_of_meals_eaten-3', 'Type_of_meals_eaten-4',
    'Type_of_fat/oil_used_in_cooking-0', 'Type_of_fat/oil_used_in_cooking-1',
    'Type_of_fat/oil_used_in_cooking-2', 'Type_of_fat/oil_used_in_cooking-3',
    'Type_of_fat/oil_used_in_cooking-4', 'Type_of_fat/oil_used_in_cooking-5',
    'Type_of_fat/oil_used_in_cooking-6', 'Type_of_fat/oil_used_in_cooking-7',
    'Type_of_fat/oil_used_in_cooking-8', 'Type_of_fat/oil_used_in_cooking-9',
    'Type_of_fat/oil_used_in_cooking-10', 'Type_of_fat/oil_used_in_cooking-11',
    'Type_of_fat/oil_used_in_cooking-12', 'Type_of_fat/oil_used_in_cooking-13',
    'Type_of_fat/oil_used_in_cooking-14', 'Type_of_fat/oil_used_in_cooking-15',
    'Type_of_fat/oil_used_in_cooking-16', 'Type_of_fat/oil_used_in_cooking-17',
    'Type_of_fat/oil_used_in_cooking-18', 'Type_of_fat/oil_used_in_cooking-19',
    'Type_of_fat/oil_used_in_cooking-20', 'Type_of_fat/oil_used_in_cooking-21',
    'Type_of_fat/oil_used_in_cooking-22', 'Type_of_fat/oil_used_in_cooking-23',
    'Type_of_sliced_bread_eaten-0', 'Type_of_sliced_bread_eaten-1',
    'Type_of_sliced_bread_eaten-2', 'Type_of_sliced_bread_eaten-3',
    'Type_of_sliced_bread_eaten-4', 'Type_of_baguette_eaten-0',
    'Type_of_baguette_eaten-1', 'Type_of_baguette_eaten-2',
    'Type_of_baguette_eaten-3', 'Type_of_baguette_eaten-4',
    'Type_of_large_bap_eaten-0', 'Type_of_large_bap_eaten-1',
    'Type_of_large_bap_eaten-2', 'Type_of_large_bap_eaten-3',
    'Type_of_bread_roll_eaten-0', 'Type_of_bread_roll_eaten-1',
    'Type_of_bread_roll_eaten-2', 'Type_of_bread_roll_eaten-3',
    'Type_of_bread_roll_eaten-4', 'Size_of_white_wine_glass_drunk-0',
    'Size_of_white_wine_glass_drunk-1', 'Size_of_white_wine_glass_drunk-2',
    'Size_of_red_wine_glass_drunk-0', 'Size_of_red_wine_glass_drunk-1',
    'Size_of_rose_wine_glass_drunk-0', 'Size_of_rose_wine_glass_drunk-1',
    'Thickness_of_butter/margarine_spread_on_sliced_bread-0',
    'Thickness_of_butter/margarine_spread_on_sliced_bread-1',
    'Thickness_of_butter/margarine_spread_on_sliced_bread-2',
    'Thickness_of_butter/margarine_spread_on_baguettes-0',
    'Thickness_of_butter/margarine_spread_on_baguettes-1',
    'Thickness_of_butter/margarine_spread_on_large_baps-0',
    'Thickness_of_butter/margarine_spread_on_large_baps-1',
    'Thickness_of_butter/margarine_spread_on_bread_rolls-0',
    'Thickness_of_butter/margarine_spread_on_bread_rolls-1',
    'Thickness_of_butter/margarine_spread_on_crackers/crispbreads-0',
    'Thickness_of_butter/margarine_spread_on_crackers/crispbreads-1',
    'Thickness_of_butter/margarine_spread_on_oatcakes-0',
    'Thickness_of_butter/margarine_spread_on_oatcakes-1',
    'Thickness_of_butter/margarine_spread_on_other_bread-0',
    'Thickness_of_butter/margarine_spread_on_other_bread-1',
    'Ingredients_in_canned_soup-0', 'Ingredients_in_canned_soup-1',
    'Ingredients_in_canned_soup-2', 'Ingredients_in_canned_soup-3',
    'Ingredients_in_canned_soup-4', 'Ingredients_in_homemade_soup-0',
    'Ingredients_in_homemade_soup-1', 'Ingredients_in_homemade_soup-2',
    'Ingredients_in_homemade_soup-3', 'Ingredients_in_homemade_soup-4',
    'Ingredients_in_homemade_soup-5', 'Values_wanted-0', 'Values_wanted-1',
    'Values_wanted-2', 'Values_wanted-3', 'Values_wanted-4', 'Values_wanted-5',
    'Values_wanted-6', 'Values_wanted-7', 'Values_wanted-8', 'Values_wanted-9',
    'Values_wanted-10', 'Values_wanted-11', 'Values_wanted-12',
    'Values_wanted-13', 'Values_wanted-14', 'Values_wanted-15',
    'Values_wanted-16', 'Values_wanted-17', 'Values_wanted-18',
    'Values_wanted-19', 'Values_wanted-20', 'Values_wanted-21',
    'Values_wanted-22', 'Values_wanted-23', 'Values_wanted-24',
    'Values_wanted-25', 'Values_wanted-26', 'Values_wanted-27',
    'Values_wanted-28', 'Values_wanted-29', 'Values_wanted-30',
    'Values_wanted-31', 'Values_wanted-32', 'Values_wanted-33',
    'Values_wanted-34', 'Values_wanted-35', 'Values_wanted-36',
    'Values_wanted-37', 'Values_wanted-38', 'Values_wanted-39',
    'Values_wanted-40', 'Values_wanted-41', 'Values_wanted-42',
    'Values_wanted-43', 'Values_wanted-44', 'Values_wanted-45',
    'Values_wanted-46', 'Values_wanted-47', 'Values_wanted-48',
    'Values_wanted-49', 'Values_wanted-50', 'Values_wanted-51',
    'Values_wanted-52', 'Values_wanted-53', 'Values_wanted-54',
    'Values_wanted-55', 'Values_wanted-56', 'Values_wanted-57',
    'Values_wanted-58', 'Values_wanted-59', 'Values_wanted-60',
    'Values_wanted-61', 'Values_wanted-62', 'Values_wanted-63',
    'Values_wanted-64', 'Values_wanted-65', 'Values_wanted-66',
    'Values_wanted-67', 'Values_wanted-68', 'Values_wanted-69',
    'Values_wanted-70', 'Values_wanted-71', 'Values_wanted-72',
    'Values_wanted-73', 'Values_wanted-74', 'Values_wanted-75',
    'Values_wanted-76', 'Values_wanted-77', 'Values_wanted-78',
    'Values_wanted-79', 'Values_wanted-80', 'Values_wanted-81',
    'Values_wanted-82', 'Values_wanted-83', 'Values_wanted-84',
    'Values_wanted-85', 'Values_wanted-86', 'Values_wanted-87',
    'Values_wanted-88', 'Values_wanted-89', 'Values_wanted-90',
    'Values_wanted-91', 'Values_wanted-92', 'Values_wanted-93',
    'Values_wanted-94', 'Values_wanted-95', 'Values_wanted-96',
    'Values_wanted-97', 'Values_wanted-98', 'Values_wanted-99',
    'Values_wanted-100', 'Values_wanted-101', 'Values_wanted-102',
    'Values_wanted-103', 'Values_wanted-104', 'Values_wanted-105',
    'Values_wanted-106', 'Values_wanted-107', 'Values_wanted-108',
    'Values_wanted-109', 'Values_wanted-110', 'Values_entered-0',
    'Values_entered-1', 'Values_entered-2', 'Values_entered-3',
    'Values_entered-4', 'Values_entered-5', 'Values_entered-6',
    'Values_entered-7', 'Values_entered-8', 'Values_entered-9',
    'Values_entered-10', 'Values_entered-11', 'Values_entered-12',
    'Values_entered-13', 'Values_entered-14', 'Values_entered-15',
    'Values_entered-16', 'Values_entered-17', 'Values_entered-18',
    'Values_entered-19', 'Values_entered-20', 'Values_entered-21',
    'Values_entered-22', 'Values_entered-23', 'Values_entered-24',
    'Values_entered-25', 'Values_entered-26', 'Values_entered-27',
    'Values_entered-28', 'Values_entered-29', 'Values_entered-30',
    'Values_entered-31', 'Values_entered-32', 'Values_entered-33',
    'Values_entered-34', 'Values_entered-35', 'Values_entered-36',
    'Values_entered-37', 'Values_entered-38', 'Values_entered-39',
    'Values_entered-40', 'Values_entered-41', 'Values_entered-42',
    'Values_entered-43', 'Values_entered-44', 'Values_entered-45',
    'Values_entered-46', 'Values_entered-47', 'Values_entered-48',
    'Values_entered-49', 'Values_entered-50', 'Values_entered-51',
    'Values_entered-52', 'Values_entered-53', 'Values_entered-54',
    'Values_entered-55', 'Values_entered-56', 'Values_entered-57',
    'Values_entered-58', 'Values_entered-59', 'Values_entered-60',
    'Values_entered-61', 'Values_entered-62', 'Values_entered-63',
    'Values_entered-64', 'Values_entered-65', 'Values_entered-66',
    'Values_entered-67', 'Values_entered-68', 'Values_entered-69',
    'Values_entered-70', 'Values_entered-71', 'Values_entered-72',
    'Values_entered-73', 'Values_entered-74', 'Values_entered-75',
    'Values_entered-76', 'Values_entered-77', 'Values_entered-78',
    'Values_entered-79', 'Values_entered-80', 'Values_entered-81',
    'Values_entered-82', 'Values_entered-83', 'Values_entered-84',
    'Values_entered-85', 'Values_entered-86', 'Values_entered-87',
    'Values_entered-88', 'Values_entered-89', 'Values_entered-90',
    'Values_entered-91', 'Values_entered-92', 'Values_entered-93',
    'Values_entered-94', 'Values_entered-95', 'Values_entered-96',
    'Values_entered-97', 'Values_entered-98', 'Values_entered-99',
    'Values_entered-100', 'Values_entered-101', 'Values_entered-102',
    'Values_entered-103', 'Values_entered-104', 'Values_entered-105',
    'Values_entered-106', 'Values_entered-107', 'Values_entered-108',
    'Values_entered-109', 'Values_entered-110',
    'Number_of_symbol_digit_matches_attempted-0', 'Breakfast_cereal_consumed-0',
    'Porridge_intake-0', 'Muesli_intake-0', 'Oat_crunch_intake-0',
    'Sweetened_cereal_intake-0', 'Plain_cereal_intake-0',
    'Bran_cereal_intake-0', 'Whole-wheat_cereal_intake-0',
    'Other_cereal_intake-0', 'Dried_fruit_added_to_cereal-0',
    'Milk_added_to_cereal-0', 'Intake_of_sugar_added_to_cereal-0',
    'Intake_of_artificial_sweetener_added_to_cereal-0', 'Type_milk_consumed-0',
    'Bread_consumed-0', 'Sliced_bread_intake-0', 'Baguette_intake-0',
    'Bap_intake-0', 'Bread_roll_intake-0', 'Naan_bread_intake-0',
    'Garlic_bread_intake-0', 'Crispbread_intake-0', 'Oatcakes_intake-0',
    'Other_bread_intake-0', 'Butter/margarine_on_bread/crackers-0',
    'Number_of_bread_slices_with_butter/margarine-0',
    'Number_of_baguettes_with_butter/margarine-0',
    'Number_of_baps_with__butter/margarine-0',
    'Number_of_bread_rolls_with__butter/margarine-0',
    'Number_of_crackers/crispbreads_with_butter/margarine-0',
    'Number_of_oatcakes_with__butter/margarine-0',
    'Number_of_other_bread_types_with__butter/margarine-0',
    'Double_crust_pastry_intake-0', 'Single_crust_pastry_intake-0',
    'Crumble_intake-0', 'Pizza_intake-0', 'Pancake_intake-0',
    'Scotch_pancake_intake-0', 'Yorkshire_pudding_intake-0',
    'Indian_snacks_intake-0', 'Croissant_intake-0', 'Danish_pastry_intake-0',
    'Scone_intake-0', 'Yogurt/ice-cream_consumers-0', 'Yogurt_intake-0',
    'Ice-cream_intake-0', 'Dessert_consumers-0', 'Milk-based_pudding_intake-0',
    'Other_milk-based_pudding_intake-0', 'Soya_dessert_intake-0',
    'Fruitcake_intake-0', 'Cake_intake-0', 'Doughnut_intake-0',
    'Sponge_pudding_intake-0', 'Cheesecake_intake-0', 'Other_dessert_intake-0',
    'Sweet_snack_consumers-0', 'Chocolate_bar_intake-0',
    'White_chocolate_intake-0', 'Milk_chocolate_intake-0',
    'Dark_chocolate_intake-0', 'Chocolate-covered_raisin_intake-0',
    'Chocolate_sweet_intake-0', 'Diet_sweets_intake-0', 'Sweets_intake-0',
    'Chocolate-covered_biscuits_intake-0', 'Chocolate_biscuits_intake-0',
    'Sweet_biscuits_intake-0', 'Cereal_bar_intake-0', 'Other_sweets_intake-0',
    'Savoury_snack_consumers-0', 'Salted_peanuts_intake-0',
    'Unsalted_peanuts_intake-0', 'Salted_nuts_intake-0',
    'Unsalted_nuts_intake-0', 'Seeds_intake-0', 'Crisp_intake-0',
    'Savoury_biscuits_intake-0', 'Cheesy_biscuits_intake-0', 'Olives_intake-0',
    'Other_savoury_snack_intake-0', 'Soup_consumers-0',
    'Powdered/instant_soup_intake-0', 'Canned_soup_intake-0',
    'Homemade_soup_intake-0', 'Starchy_food_consumers-0',
    'White_pasta_intake-0', 'Wholemeal_pasta_intake-0', 'White_rice_intake-0',
    'Brown_rice_intake-0', 'Sushi_intake-0', 'Snackpot_intake-0',
    'Couscous_intake-0', 'Other_grain_intake-0', 'Cheese_consumers-0',
    'Low_fat_hard_cheese_intake-0', 'Hard_cheese_intake-0',
    'Soft_cheese_intake-0', 'Blue_cheese_intake-0',
    'Low_fat_cheese_spread_intake-0', 'Cheese_spread_intake-0',
    'Cottage_cheese_intake-0', 'Feta_intake-0', 'Mozzarella_intake-0',
    "Goat's_cheese_intake-0", 'Other_cheese_intake-0', 'Egg_consumers-0',
    'Whole_egg_intake-0', 'Omelette_intake-0', 'Eggs_in_sandwiches_intake-0',
    'Scotch_egg_intake-0', 'Other_egg_intake-0', 'Meat_consumers-0',
    'Sausage_intake-0', 'Beef_intake-0', 'Pork_intake-0', 'Lamb_intake-0',
    'Crumbed_or_deep-fried_poultry_intake-0', 'Poultry_intake-0',
    'Bacon_intake-0', 'Ham_intake-0', 'Liver_intake-0', 'Other_meat_intake-0',
    'Fat_removed_from_meat-0', 'Skin_removed_from_poultry-0', 'Fish_consumer-0',
    'Tinned_tuna_intake-0', 'Oily_fish_intake-0', 'Breaded_fish_intake-0',
    'Battered_fish_intake-0', 'White_fish_intake-0', 'Prawns_intake-0',
    'Lobster/crab_intake-0', 'Shellfish_intake-0', 'Other_fish_intake-0',
    'Vegetarian_alternatives_intake-0', 'Vegetarian_sausages/burgers_intake-0',
    'Tofu_intake-0', 'Quorn_intake-0', 'Other_vegetarian_alternative_intake-0',
    'Spreads/sauces_consumers-0', 'No_fat_for_cooking-0',
    'Vegetable_consumers-0', 'Baked_bean_intake-0', 'Pulses_intake-0',
    'Fried_potatoes_intake-0', 'Boiled/baked_potatoes_intake-0',
    'Butter/margarine_added_to_potatoes-0', 'Mashed_potato_intake-0',
    'Mixed_vegetable_intake-0', 'Vegetable_pieces_intake-0',
    'Coleslaw_intake-0', 'Side_salad_intake-0', 'Avocado_intake-0',
    'Broad_bean_intake-0', 'Green_bean_intake-0', 'Beetroot_intake-0',
    'Broccoli_intake-0', 'Butternut_squash_intake-0', 'Cabbage/kale_intake-0',
    'Carrot_intake-0', 'Cauliflower_intake-0', 'Celery_intake-0',
    'Courgette_intake-0', 'Cucumber_intake-0', 'Garlic_intake-0',
    'Leek_intake-0', 'Lettuce_intake-0', 'Mushroom_intake-0', 'Onion_intake-0',
    'Parsnip_intake-0', 'Pea_intake-0', 'Sweet_pepper_intake-0',
    'Spinach_intake-0', 'Sprouts_intake-0', 'Sweetcorn_intake-0',
    'Sweet_potato_intake-0', 'Fresh_tomato_intake-0', 'Tinned_tomato_intake-0',
    'Turnip/swede_intake-0', 'Watercress_intake-0', 'Other_vegetables_intake-0',
    'Fruit_consumers-0', 'Stewed_fruit_intake-0', 'Prune_intake-0',
    'Dried_fruit_intake-0', 'Mixed_fruit_intake-0', 'Apple_intake-0',
    'Banana_intake-0', 'Berry_intake-0', 'Cherry_intake-0',
    'Grapefruit_intake-0', 'Grape_intake-0', 'Mango_intake-0',
    'Melon_intake-0', 'Orange_intake-0', 'Satsuma_intake-0',
    'Peach/nectarine_intake-0', 'Pear_intake-0', 'Pineapple_intake-0',
    'Plum_intake-0', 'Other_fruit_intake-0',
    'When_diet_questionnaire_completed-0',
    'When_diet_questionnaire_started-0',
    'Ever_had_breast_cancer_screening_/_mammogram-0',
    'Years_since_last_breast_cancer_screening_/_mammogram-0',
    'Ever_had_cervical_smear_test-0',
    'Years_since_last_cervical_smear_test-0',
    'Number_of_live_births-0',
    'Birth_weight_of_first_child-0',
    'Ever_used_hormone-replacement_therapy_(HRT)-0',
    'Age_at_hysterectomy-0',
    'Bilateral_oophorectomy_(both_ovaries_removed)-0',
    'Weight,_manual_entry-0',
    'Year_immigrated_to_UK_(United_Kingdom)-0',
    'Number_of_stillbirths-0',
    'Number_of_spontaneous_miscarriages-0',
    'Number_of_pregnancy_terminations-0',
    'Age_of_primiparous_women_at_birth_of_child-0',
    'Age_at_bilateral_oophorectomy_(both_ovaries_removed)-0',
    'Digits_entered_correctly-0', 'Digits_entered_correctly-1',
    'Digits_entered_correctly-2', 'Digits_entered_correctly-3',
    'Digits_entered_correctly-4', 'Digits_entered_correctly-5',
    'Digits_entered_correctly-6', 'Digits_entered_correctly-7',
    'Digits_entered_correctly-8', 'Digits_entered_correctly-9',
    'Digits_entered_correctly-10', 'Digits_entered_correctly-11',
    'Digits_entered_correctly-12', 'Digits_entered_correctly-13',
    'Digits_entered_correctly-14', 'Digits_entered_correctly-15',
    'Qualifications-0', 'Qualifications-1', 'Qualifications-2',
    'Qualifications-3', 'Qualifications-4', 'Qualifications-5',
    'Current_employment_status-1', 'Current_employment_status-2',
    'Current_employment_status-3', 'Current_employment_status-4',
    'Current_employment_status-5', 'Current_employment_status-6',
    'Transport_type_for_commuting_to_job_workplace-1',
    'Transport_type_for_commuting_to_job_workplace-2',
    'Transport_type_for_commuting_to_job_workplace-3',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-1',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-2',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-3',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-4',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-5',
    'Manic/hyper_symptoms-0', 'Manic/hyper_symptoms-1',
    'Manic/hyper_symptoms-2', 'Manic/hyper_symptoms-3',
    'Impedance_of_whole_body,_manual_entry-0',
    'Impedance_of_leg,_manual_entry_(right)-0',
    'Impedance_of_leg,_manual_entry_(left)-0',
    'Impedance_of_arm,_manual_entry_(right)-0',
    'Impedance_of_arm,_manual_entry_(left)-0',
    'Vitamin_and/or_mineral_supplement_use-0',
    'Vitamin_and/or_mineral_supplement_use-1',
    'Vitamin_and/or_mineral_supplement_use-2',
    'Vitamin_and/or_mineral_supplement_use-3',
    'Vitamin_and/or_mineral_supplement_use-4',
    'Vitamin_and/or_mineral_supplement_use-5',
    'Vitamin_and/or_mineral_supplement_use-6',
    'Vitamin_and/or_mineral_supplement_use-7',
    'Vitamin_and/or_mineral_supplement_use-8',
    'Vitamin_and/or_mineral_supplement_use-9',
    'Vitamin_and/or_mineral_supplement_use-10',
    'Vitamin_and/or_mineral_supplement_use-11',
    'Vitamin_and/or_mineral_supplement_use-12',
    'Vitamin_and/or_mineral_supplement_use-13',
    'Vitamin_and/or_mineral_supplement_use-14',
    'Vitamin_and/or_mineral_supplement_use-15',
    'Vitamin_and/or_mineral_supplement_use-16',
    'Vitamin_and/or_mineral_supplement_use-17',
    'Vitamin_and/or_mineral_supplement_use-18',
    'Vitamin_and/or_mineral_supplement_use-19',
    'Vitamin_and/or_mineral_supplement_use-20',
    'Liquid_used_to_make_porridge-0', 'Liquid_used_to_make_porridge-1',
    'Type_of_yogurt_eaten-0', 'Type_of_yogurt_eaten-1',
    'Number_of_fluid_intelligence_questions_attempted_within_time_limit-0',
    'Added_salt_to_food-0', 'Vitamin_supplement_user-0',
    'Attempted_fluid_intelligence_(FI)_tes-2',
    'FI1_:_numeric_addition_test-0',
    'FI2_:_identify_largest_number-0',
    'FI3_:_word_interpolation-0',
    'FI4_:_positional_arithmetic-0',
    'FI5_:_family_relationship_calculation-0',
    'FI6_:_conditional_arithmetic-0',
    'FI7_:_synonym-0', 'FI8_:_chained_arithmetic-0',  # < 30000 sbjs
    'FI9_:_concept_interpolation-0',  # < 15000 sbjs
    'FI10_:_arithmetic_sequence_recognition-0',  # < 10000 sbjs
    'FI11_:_antonym-0',  # < 5000 sbjs
    'FI12_:_square_sequence_recognition-0',  # < 5000 sbjs
    'FI13_:_subset_inclusion_logic-0',  # < 1500 sbjs
    'Ethnic_background-0',  # weird tree structure and politically shakey
    'Spirometry_method-0', 'Spirometry_device_serial_number-0',
]

CNFD = healthy_img.drop(
    drop_col, axis=1, inplace=False).copy()
logger.info('Dropped uninteresting columns.')

# %%
# 3) Pre-cleaning

# replace "()"
CNFD.columns = [x.replace("(", "") for x in CNFD.columns]  # replace "("
CNFD.columns = [x.replace(")", "") for x in CNFD.columns]  # replace ")"
logger.info('Replaced "(" and ")" with space in column names.')

# Add date-time column as decimal
CNFD['Brain_MRI_sign-off_timestamp-0'] = pd.to_datetime(
    CNFD['Brain_MRI_sign-off_timestamp-0'])
CNFD['Brain_MRI_sign-off_timestamp-0_decimal'] = (
    CNFD['Brain_MRI_sign-off_timestamp-0'].dt.hour
    + (CNFD['Brain_MRI_sign-off_timestamp-0'].dt.minute / 60)
    + (CNFD['Brain_MRI_sign-off_timestamp-0'].dt.second / 3600)
)
logger.info('Replaced brain scan sign off stamp with decimal time.')

# Drop non-number columns
CNFD = CNFD.select_dtypes([np.number])
CNFD_clean = CNFD.copy()
logger.info('Only kept number columns.')

# %%
# 5) Clean Confounds

# average columns
hgs = ['Hand_grip_strength_left-0', 'Hand_grip_strength_right-0']
CNFD_clean['HGS_mean_left_right'] = CNFD[hgs].mean(axis=1)
CNFD_clean['mean_Height-0'] = CNFD[
    ['Standing_height-0', 'Height-0']].mean(axis=1)
CNFD_clean.drop(['Standing_height-0', 'Height-0'], axis=1, inplace=True)
CNFD_clean['mean_Seated_Height-0'] = CNFD[
    ['Seated_height-0', 'Sitting_height-0']].mean(axis=1)
CNFD_clean.drop(['Seated_height-0', 'Sitting_height-0'], axis=1, inplace=True)
sys = [
    'Systolic_blood_pressure,_manual_reading-0',
    'Systolic_blood_pressure,_manual_reading-1',
    'Systolic_blood_pressure,_automated_reading-0',
    'Systolic_blood_pressure,_automated_reading-1',
    ]
CNFD_clean['mean_Systolic_blood_pressure-0'] = CNFD[sys].mean(axis=1)
CNFD_clean.drop(sys, axis=1, inplace=True)
dias = [
    'Diastolic_blood_pressure,_manual_reading-0',
    'Diastolic_blood_pressure,_manual_reading-1',
    'Diastolic_blood_pressure,_automated_reading-0',
    'Diastolic_blood_pressure,_automated_reading-1',
    ]
CNFD_clean['mean_Diastolic_blood_pressure-0'] = CNFD[dias].mean(axis=1)
CNFD_clean.drop(dias, axis=1, inplace=True)
pulse = [
    'Pulse_rate_during_blood-pressure_measurement-0',
    'Pulse_rate_during_blood-pressure_measurement-1',
    'Pulse_rate,_automated_reading-0', 'Pulse_rate,_automated_reading-1',
    'Pulse_rate-0',
    ]
CNFD_clean['mean_Pulse_rate-0'] = CNFD[pulse].mean(axis=1)
CNFD_clean.drop(pulse, axis=1, inplace=True)
fvc = [
    'Forced_vital_capacity_FVC-0',
    'Forced_vital_capacity_FVC-1',
    'Forced_vital_capacity_FVC-2',
    ]
CNFD_clean['mean_Forced_vital_capacity_FVC-0'] = CNFD[fvc].mean(axis=1)
CNFD_clean.drop(fvc, axis=1, inplace=True)
fev = [
    'Forced_expiratory_volume_in_1-second_FEV1-0',
    'Forced_expiratory_volume_in_1-second_FEV1-1',
    'Forced_expiratory_volume_in_1-second_FEV1-2',
    ]
CNFD_clean[
    'mean_Forced_expiratory_volume_in_1-second_FEV1-0'] = CNFD[fev].mean(axis=1)
CNFD_clean.drop(fev, axis=1, inplace=True)
pef = [
    'Peak_expiratory_flow_PEF-0',
    'Peak_expiratory_flow_PEF-1',
    'Peak_expiratory_flow_PEF-2',
    ]
CNFD_clean['mean_Peak_expiratory_flow_PEF-0'] = CNFD[pef].mean(axis=1)
CNFD_clean.drop(pef, axis=1, inplace=True)
logger.info('Averaged columns where needed to sumamrize.')

# replace negative value subjects with NaN
neg = [
    'Time_employed_in_main_current_job-0',
    'Length_of_working_week_for_main_job-0',
    'Age_completed_full_time_education-0',
    'Age_started_oral_contraceptive_pill-0',
    'Age_when_last_used_oral_contraceptive_pill-0',
    'Age_started_hormone-replacement_therapy_HRT-0',
    'Age_last_used_hormone-replacement_therapy_HRT-0',
    'Country_of_birth_UK/elsewhere-0',
    'Handedness_chirality/laterality-0',
    'Maximum_digits_remembered_correctly-0',
    'Longest_period_of_depression-0',
    'Number_of_depression_episodes-0',
    'Longest_period_of_unenthusiasm_/_disinterest-0',
    'Number_of_unenthusiastic/disinterested_episodes-0',
    'Length_of_longest_manic/irritable_episode-0',
    'Job_involves_mainly_walking_or_standing-0',
    'Job_involves_heavy_manual_or_physical_work-0',
    'Job_involves_shift_work-0',
    'Comparative_body_size_at_age_10-0',
    'Comparative_height_size_at_age_10-0',
    'Frequency_of_depressed_mood_in_last_2_weeks-0',
    'Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-0',
    'Frequency_of_tenseness_/_restlessness_in_last_2_weeks-0',
    'Frequency_of_tiredness_/_lethargy_in_last_2_weeks-0',
    'Job_involves_night_shift_work-0',
    'Happiness-0',
    'Breastfed_as_a_baby-0',
    'Adopted_as_a_child-0',
    'Part_of_a_multiple_birth-0',
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
    'Severity_of_manic/irritable_episodes-0',
    'Age_at_first_live_birth-0',
    'Age_at_last_live_birth-0',
    ]

for neg_col in neg:
    # print(neg_col)
    condition = (CNFD_clean[neg_col] < 0)
    # print(condition.sum())
    CNFD_clean.loc[condition, neg_col] = np.nan
logger.info('Replaced negative entries with NaN values b/c not meaningful.')

# replace subjects based on special conditions with NaN
cols = [
    'Frequency_of_travelling_from_home_to_job_workplace-0',
    'Distance_between_home_and_job_workplace-0',
    ]
for col in cols:
    CNFD_clean.loc[CNFD_clean[col] == -10, col] = 0
    CNFD_clean.loc[CNFD_clean[col] < 0, col] = np.nan
cols = [
    'Current_employment_status-0',  # keep -7
    'Transport_type_for_commuting_to_job_workplace-0',
    'Illness,_injury,_bereavement,_stress_in_last_2_years-0',
    ]
for col in cols:
    CNFD_clean.loc[CNFD_clean[col] == -3, col] = np.nan

# replace 0 with NaN
cols = [
    'Duration_to_complete_numeric_path_trail_#1-0',  # unit: deciseconds
    'Duration_to_complete_alphanumeric_path_trail_#2-0',
]
for col in cols:
    CNFD_clean.loc[CNFD_clean[col] == 0.0, col] = np.nan
logger.info('Replace subjects based on special conditions with NaN.')

# replace with NaN in errors if: errors==0 AND duration==NaN (did not do task)
dur_nan_idx = CNFD_clean[
    CNFD_clean['Duration_to_complete_numeric_path_trail_#1-0'].isnull()].index
err_null_idx = CNFD_clean[
    CNFD_clean["Total_errors_traversing_numeric_path_trail_#1-0"] == 0].index
inter = dur_nan_idx.intersection(err_null_idx)
CNFD_clean.loc[
    inter, "Total_errors_traversing_numeric_path_trail_#1-0"] = np.nan

dur_nan_idx = CNFD_clean[
    CNFD_clean['Duration_to_complete_alphanumeric_path_trail_#2-0'].isnull()
    ].index
err_null_idx = CNFD_clean[
    CNFD_clean['Total_errors_traversing_alphanumeric_path_trail_#2-0'] == 0
    ].index
inter = dur_nan_idx.intersection(err_null_idx)
CNFD_clean.loc[
    inter, 'Total_errors_traversing_alphanumeric_path_trail_#2-0'] = np.nan
logger.info(
    'Replace with NaN in TMT errors if: errors==0 AND duration==NaN '
    '(did not do task).')

# %%
# 6) Check for outliers

cont_cols = [
    'Hand_grip_strength_left-0', 'Hand_grip_strength_right-0',
    'HGS_mean_left_right', 'Brain_MRI_sign-off_timestamp-0_decimal',
    'Waist_circumference-0', 'Hip_circumference-0',
    'mean_Height-0', 'mean_Seated_Height-0', 'Weight-0',
    'mean_Systolic_blood_pressure-0', 'mean_Diastolic_blood_pressure-0',
    'mean_Pulse_rate-0', 'Time_employed_in_main_current_job-0',
    'Length_of_working_week_for_main_job-0',
    'Frequency_of_travelling_from_home_to_job_workplace-0',
    'Distance_between_home_and_job_workplace-0',
    'Age_completed_full_time_education-0',
    'Age_at_first_live_birth-0', 'Age_at_last_live_birth-0',
    'Age_started_oral_contraceptive_pill-0',
    'Age_when_last_used_oral_contraceptive_pill-0',
    'mean_Forced_vital_capacity_FVC-0',
    'mean_Forced_expiratory_volume_in_1-second_FEV1-0',
    'mean_Peak_expiratory_flow_PEF-0',
    'Age_started_hormone-replacement_therapy_HRT-0',
    'Age_last_used_hormone-replacement_therapy_HRT-0',
    'Heel_bone_mineral_density_BMD_T-score,_automated_left-0',
    'Heel_bone_mineral_density_BMD_T-score,_automated_right-0',
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_left-0',
    'Heel_bone_mineral_density_BMD_T-score,_manual_entry_right-0',
    'Maximum_digits_remembered_correctly-0',
    'Longest_period_of_depression-0', 'Number_of_depression_episodes-0',
    'Longest_period_of_unenthusiasm_/_disinterest-0',
    'Number_of_unenthusiastic/disinterested_episodes-0',
    'Length_of_longest_manic/irritable_episode-0',
    'Duration_to_complete_numeric_path_trail_#1-0',
    'Total_errors_traversing_numeric_path_trail_#1-0',
    'Duration_to_complete_alphanumeric_path_trail_#2-0',
    'Total_errors_traversing_alphanumeric_path_trail_#2-0',
    'Number_of_puzzles_correctly_solved-0',
    'Fluid_intelligence_score-0', 'Mean_time_to_correctly_identify_matches-0',
    'Number_of_word_pairs_correctly_associated-0', 'Body_mass_index_BMI-0',
    'Number_of_puzzles_correct-0', 'Pulse_wave_Arterial_Stiffness_index-0',
    'Body_fat_percentage-0', 'Whole_body_fat_mass-0',
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
    'Trunk_fat_percentage-0', 'Trunk_fat_mass-0', 'Trunk_fat-free_mass-0',
    'Trunk_predicted_mass-0', 'Number_of_symbol_digit_matches_made_correctly-0',
    'Age-0',
]

# Replace subject entries deviating more than
# "std_limit" SDs from column distribution with NaN
for col in cont_cols:
    num_outliers = len(
        CNFD_clean.loc[
            np.abs(zscore(CNFD_clean[col], nan_policy='omit')) > std_limit,
            col].index)
    CNFD_clean.loc[
        np.abs(zscore(CNFD_clean[col], nan_policy='omit')) > std_limit,
        col] = np.nan
    logger.info(f'Replaced {num_outliers} outliers in {col} with NaN value.')

# %% 7) Convert & save dataframe as .jay

CNFD_clean.reset_index(level=0, inplace=True)
CNFD_clean_dt = dt.Frame(CNFD_clean)
CNFD_clean_dt.to_jay(
    (phenotype_dir / '40_allUKB_reduced_cleaned_exICD10-V-VI-stroke_IMG.jay'
     ).as_posix())
logger.info(
    'Reduced and cleaned all UKB potentially usable phenotype .jay was saved.')
# %%
