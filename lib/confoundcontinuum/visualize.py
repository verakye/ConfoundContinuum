from pathlib import Path
from confoundcontinuum.logging import logger
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------#
# Prediction visualization
# -----------------------------------------------------------------------------#


def visualize_predictions(
        y_true, y_pred, color=None, error_measures=None,
        set_axes_labels=True, font_size=20, fig_fname=None, ymin=5, ymax=65,
        xmin=5, xmax=65, x_label=None, y_label=None):
    """
    Hexagon density plot including x- and y-axis distributions and prediction
    error measures as well as the hypothetical perfect prediction line and the
    regression line between true and predicted targets.

    Parameters
    ----------
    y_true : DataFrame Series
        Series object containing the true target values per subject.
    y_pred : DataFrame Series
        Series object containing the predicted target values per subject.
    color : str
        Color for plot in hex colour. Default "#0077B6" (blue).
    error_measures : Dict
        Dictionary containing MAE, RMSE, R^2 and r as key-value pairs (in that
        order!).
    set_axes_labels: Boolean
        Whether to set the axes labels. Defaults to True.
    font_size: int
        font-size for labels. Default: 20.
    fig_fname : None or str/Path
        If None, figure will not be saved. Otherwise pass a string with the
        directory and the name to save the figure. (.pdf recommended)

    Returns
    -------
    figure: hexbin
    """
    # set defaults and make checks
    if color is None:
        color = "#0077B6"
    if error_measures is None:
        error_measures = {'MAE': '', 'RMSE': '', 'R2': '', 'pearsonr': ''}
        logger.info('No error measures were provided.')
    if fig_fname is None:
        logger.info(
            'No directory to save the figure was provided. Figure will not be'
            'saved.')
    elif fig_fname is not None:
        fig_fname = Path(fig_fname) if isinstance(fig_fname, str) else fig_fname

    # set margins
    ymin = ymin
    ymax = ymax
    xmin = xmin
    xmax = xmax

    # set axes labels
    if x_label is None:
        x_label = 'True hand grip strength (kg)'
    if y_label is None:
        y_label = 'Predicted hand grip strength (kg)'

    sns.set_style('white')
    h = sns.jointplot(
        x=y_true, y=y_pred, kind="hex", color=color)
    sns.regplot(
        x=y_true, y=y_pred, ax=h.ax_joint, scatter=False, ci=99,
        line_kws={'lw': 1}, color=color)  # "#5d5d60"
    h.plot_marginals(sns.histplot, kde=True, color=color)

    # sns.distplot(y_pred,ax=h.ax_marg_y,color='r',vertical=True)
    # xmin, xmax = h.ax_joint.get_xlim()
    h.ax_joint.set(ylim=[ymin, ymax])
    h.ax_joint.set(xlim=[xmin, xmax])

    # plt.plot(y_true, y_true, color='k')  # perfect "prediction"
    # plt.ylim([ymin, 2])

    # Error measures
    text = (
        ' MAE = ' + str(error_measures['MAE']) +
        # '\n RMSE = ' + str(error_measures['RMSE']) +
        '\n $R^{2}$ = ' + str(error_measures['R2']) + '\n '
        r"$\bf{r = " + str(error_measures['pearsonr']) + "}$")
    if set_axes_labels:
        h.set_axis_labels(x_label, y_label, fontsize=font_size)  # was 25 before
    else:
        h.set_axis_labels(None)
    for tick in h.ax_joint.get_xticklabels():
        tick.set_fontsize(font_size)
    for tick in h.ax_joint.get_yticklabels():
        tick.set_fontsize(font_size)
    # h.ax_marg_x.set_title(f'Actual vs Predicted {y}')
    plt.text(xmin+2.5, ymax-0.05, text,
             verticalalignment='top', horizontalalignment='left',
             fontsize=font_size, fontname='Arial')
    # plt.tight_layout()
    # plt.axis('scaled')
    if fig_fname is not None:
        plt.savefig(fig_fname.as_posix(), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def colored_scatter(x, y, c=None, gridsize=None, vmin=None, vmax=None):
    def hexbin(*args, **kwargs):
        args = (x, y)
        cmap = sns.light_palette(c, as_cmap=True)
        # norm = plt.Normalize(df[col_x].min(), df[col_x].max())
        if c is not None:
            kwargs['cmap'] = cmap
        if gridsize is not None:
            kwargs['gridsize'] = gridsize
        if vmin and vmax is not None:
            kwargs['vmin'] = vmin
            kwargs['vmax'] = vmax
        # kwargs['norm'] = norm
        plt.hexbin(*args, mincnt=1, **kwargs)

    return hexbin
