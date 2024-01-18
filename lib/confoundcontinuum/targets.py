from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import zscore

from confoundcontinuum.logging import logger, raise_error


# -----------------------------------------------------------------------------#
# UK BioBank
# -----------------------------------------------------------------------------#


def parse_phenotypes_ukbb(
        in_ukbb_dir, out_phen_dir, excludeICD10=None, incats=None, inhdrs=None):
    """
    Parses from ukbb tsv all ukbb available motor phenotypes by specifying the
    required categories (incat) and/or datafields (inhdr) and excluding subjects
    under specific ICD10 criteria (excludeICD10), using ukbb_parser.
    Datafield 53, containing the dates of the respective assessment is always
    included to allow for control of the parsed subjects.

    Parameters
    ----------
    in_ukbb_dir: str or Path
        Directory of tsv file as downloaded from the UKBB containing the
        phenotype data.
    out_phen_dir: str or Path
        Directory where to save the parsed phenotype .csv and .html file
    excludeICD10 : list
        list of strings providing the ICD10 code of criteria to be excluded. A
        string of a range, e.g. "I60-I60" can be passed.
    incat : list
        list of category IDs to parse. Provided as integers
    inhdr : list
        list of datafiled IDs to parse. Provided as integers

    Returns
    -------
    parse_process_result.stdout.decode(): str
        Output/error information of the parsing command subprocess
    """
    if in_ukbb_dir is Path:
        in_ukbb_dir = in_ukbb_dir.as_posix()
    if out_phen_dir is Path:
        out_phen_dir = out_phen_dir.as_posix()

    parse_command = [
        "ukbb_parser", "parse",
        "--incsv", f"{in_ukbb_dir}",
        "--out", f"{out_phen_dir}",
        "--long_names",
        "--fillna", "NaN"]
    if excludeICD10 is not None:
        for excon in excludeICD10:
            parse_command.extend(["--excon", f"{excon}"])
    if inhdrs is None:
        inhdrs = [53]  # always parse the date of the assessment centers
    elif inhdrs is not None:
        inhdrs = inhdrs + [53]
        for inhdr in inhdrs:
            parse_command.extend(["--inhdr", f"{inhdr}"])
    if incats is not None:
        for incat in incats:
            parse_command.extend(["--incat", f"{incat}"])

    parse_process_result = subprocess.run(
        parse_command,
        stdout=subprocess.PIPE,
        check=True)  # throw an exception when process returns nonzero exit code

    return parse_process_result.stdout.decode()  # check out/err


def shape_phenotypes_ukbb(
        in_df, columns2remove=None, session2keep=None, keeprun=False):
    # set subject-ID as index
    in_df['eid'] = 'sub-' + in_df['eid'].astype(str)
    in_df = in_df.set_index(['eid'])
    in_df.index.name = 'SubjectID'

    # modify age and sex data and define further session specificities
    if session2keep == 'ses-0':
        in_df.rename(columns={'Age1stVisit': 'Age-0.0'}, inplace=True)
        age_remove_cols = ['AgeRepVisit', 'AgeAtScan', 'AgeAt2ndScan']
        regex = '-0'
    elif session2keep == 'ses-1':
        in_df.rename(
            columns={'Sex-0.0': 'Sex-1.0', 'AgeRepVisit': 'Age-1.0'},
            inplace=True)
        age_remove_cols = ['Age1stVisit', 'AgeAtScan', 'AgeAt2ndScan']
        regex = '-1'
    elif session2keep == 'ses-2':
        in_df.rename(
            columns={'Sex-0.0': 'Sex-2.0', 'AgeAtScan': 'Age-2.0'},
            inplace=True)
        age_remove_cols = ['Age1stVisit', 'AgeRepVisit', 'AgeAt2ndScan']
        regex = '-2'
    elif session2keep == 'ses-3':
        in_df.rename(
            columns={'Sex-0.0': 'Sex-3.0', 'AgeAt2ndScan': 'Age-3.0'},
            inplace=True)
        age_remove_cols = ['Age1stVisit', 'AgeRepVisit', 'AgeAtScan']
        regex = '-3'

    if columns2remove is None:
        columns2remove = ([
            'NP_controls_1', 'NP_controls_2', 'CNS_controls_1',
            'CNS_controls_2', 'Race', 'YearsOfEducation', 'ISCED',
            # 'Date_of_attending_assessment_centre-0.0',
            # 'Date_of_attending_assessment_centre-1.0',
            # 'Date_of_attending_assessment_centre-2.0',
            # 'Date_of_attending_assessment_centre-3.0'
        ]
            + age_remove_cols)
    in_df.drop(columns2remove, axis=1, inplace=True)

    if session2keep is not None:
        # keep data from specified session
        in_df = in_df.filter(regex=regex).copy()

        if keeprun:
            # change col names without session no but with run within session
            in_df.columns = [
                x.split('.')[0][0:-1]+x.split('.')[1]
                for x in list(in_df.columns)]
        else:
            # change column names without session number
            in_df.columns = [x.split('.')[0][0:-2] for x in list(in_df.columns)]

    return in_df


def modify_hgs_ukbb(in_df):
    # add mean
    HGS = in_df[
        ['Hand_grip_strength_(left)', 'Hand_grip_strength_(right)']].copy()
    in_df['HGS_mean_left_right'] = HGS.mean(axis=1)
    logger.debug('Added mean HGS (left/right) as HGS column.')

    return in_df


def dropnan_ukbb(in_df, min_nonans=25000):
    """
    Drop all rows which have a NaN entry in any column. Only keep a column when
    it has more than "min_nonans" non-NaN rows. Set "min_nonans" to 0 to not
    apply a threshold.
    """
    # apply threshold (filter columns)
    in_df = in_df.loc[:, (~in_df.isnull()).sum() > min_nonans].copy()
    # drop NaN rows
    nan = in_df[in_df.isnull().any(axis=1)].index
    in_df.drop(labels=nan, inplace=True)

    return in_df


def remove_outliers_ukbb(
        in_df, std_limit, non_number_cols=None, drop_OL=True,
        include_ageOL=True):
    # Set 'sex' as default non_number_column
    if non_number_cols is None:
        non_number_cols = ['Sex']
    else:
        non_number_cols = ['Sex'] + non_number_cols

    # Check if non_number_cols in df and drop (save in other df)
    non_number_cols_confirmed = [
        col for col in non_number_cols if col in in_df.columns]
    non_number_cols_confirmed = list(set(non_number_cols_confirmed))  # rm dupl.
    # store non_number_cols in separate dataframe and drop in in_df
    non_number_df = in_df[non_number_cols_confirmed].copy()
    in_df.drop(non_number_cols_confirmed, axis=1, inplace=True)

    # Get outlier rows caused by age
    if include_ageOL is True:
        if 'Age' not in in_df.columns:
            raise_error('Provide an age column to identify age outliers.')
        else:
            outliers_age = in_df[
                (~(np.abs(zscore(in_df['Age'])) < std_limit))].copy()
            outlier_list_age = outliers_age.index.unique().tolist()

    # Get outlier rows caused by non-age columns
    if 'Age' in in_df.columns:
        considered_cols = in_df.columns.to_list()
        considered_cols.remove('Age')
    else:
        considered_cols = in_df.columns.to_list()
    outliers = in_df[
        (~(np.abs(in_df[considered_cols].apply(zscore)) < std_limit)
         ).any(axis=1)].copy()
    outlier_list = outliers.index.unique().tolist()  # outlier indices (SbjID)

    # Outlier overlap age-non_age outliers
    if 'outlier_list_age' in locals():
        intersection_age_nonAge_OL = (list(
            set(outlier_list_age) & set(outlier_list)))
        logger.info(
            f'There is an overlap of {len(intersection_age_nonAge_OL)} rows '
            'between age and non-age outliers.')

    # Drop outlier subjects
    if drop_OL is True:
        # Drop outlier subjects in any variable (except for age and no num cols)
        in_df.drop(labels=outlier_list, axis=0, inplace=True)
        logger.info(
            f'{len(outliers)} non age related outlier rows were dropped.')
        # Drop age outliers
        if 'outliers_age' in locals():
            outlier_drop_list_age = list(
                set(outlier_list_age) - set(intersection_age_nonAge_OL))
            in_df.drop(labels=outlier_drop_list_age, axis=0, inplace=True)
            logger.info(
                f'{len(outlier_drop_list_age)} age driven outlier rows were '
                'dropped after consideration of overlapping rows.')
    else:
        logger.info(
            f'{len(outliers)} non age related outlier rows were identified. '
            'Set drop_OL=True to drop them.')
        if 'outliers_age' in locals():
            logger.info(
                f'{len(outliers_age)} age driven outlier rows were identified, '
                'set drop_OL=True to drop them.')

    # Add unchanged non_number_columns
    in_df = in_df.join(other=non_number_df).copy()

    if include_ageOL is True:
        return in_df, outlier_list, outlier_list_age, intersection_age_nonAge_OL
    else:
        return in_df, outlier_list


def check_sex_balance_ukbb(in_df):
    num_male = in_df[in_df['Sex'] == 1.0].shape[0]
    num_female = in_df[in_df['Sex'] == 0.0].shape[0]
    per_male = num_male/(num_male+num_female)
    per_female = num_female/(num_male+num_female)

    logger.info(
        f'{num_male} men ({per_male:.2f}%), '
        f'{num_female} women ({per_female:.2f}%).')

    return per_male, per_female
