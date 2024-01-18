from sklearn.metrics import make_scorer
from scipy.stats import pearsonr, spearmanr
import numpy as np
from confoundcontinuum.logging import logger


# -----------------------------------------------------------------------------#
# Helper functions / Calculations
# -----------------------------------------------------------------------------#

def heuristic_C(data_df=None):
    """
    Calculate the heuristic C for linearSVR (Joachims 2002).

    Returns
    -------
    HP : dict
        Dictionary containing the theoretically calculated hyperparameter C
        for a linear SVM as a float (value) and the name of the HP (C) as a key.
        Returns a dict for better usability with HeuristicWrapper class.
    """

    if data_df is None:
        logger.error('No data was provided.')

    C = 1/np.mean(np.sqrt((data_df**2).sum(axis=1)))
    # Formular Kaustubh: C = 1/mean(sqrt(rowSums(data^2)))

    HP = {
        'C': C,
    }
    return HP


# -----------------------------------------------------------------------------#
# Add pearson correlation as a valid sklearn scorer
# -----------------------------------------------------------------------------#


def pear_corr(y_true, y_pred):
    cor, _ = pearsonr(y_true, y_pred)
    return cor


def spear_corr(y_true, y_pred):
    cor, _ = spearmanr(y_true, y_pred)
    return cor


pearson_scorer = make_scorer(pear_corr)
spearman_scorer = make_scorer(spear_corr)

# -----------------------------------------------------------------------------#
