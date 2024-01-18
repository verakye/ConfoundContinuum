from functools import partial
from pathlib import Path, PurePath
import io
import shutil

from confoundcontinuum.logging import configure_logging, log_versions
from confoundcontinuum.logging import logger, raise_error

import numpy as np
import pandas as pd
import re
from scipy.stats.mstats import winsorize

import requests

from nilearn import datasets
from nilearn import image
from nilearn import masking
from nilearn.datasets.utils import _get_dataset_descr

import nest_asyncio
nest_asyncio.apply()
import datalad.api as dl  # noqa E402

configure_logging()
log_versions()

"""
A dictionary containing all supported atlases and their respective valid
parameters.
"""

_available_atlases = {
    # cortical
    'Schaefer': {
        'valid_nrois': np.arange(100, 1100, 100).tolist(),
        'valid_yeonetworks': [7, 17],
        'valid_resolution': [1, 2],
        'kind': ['cortical', 'nifti'],
    },
    # subcortical
    'Tian': {
        'valid_scales': [1, 2, 3, 4],
        'valid_magenticfield': [3, 7],
        'valid_space': ['MNI152_nonlinear_6th_generation',
                        'MNI152_nonlinear_2009cAsym'],
        'valid_resolution': [1, 1.6, 2],
        'kind': ['subcortical', 'nifti']
    },
    # cerebellar
    'SUIT': {
        'valid_space': ['MNI', 'SUIT'],
        'valid_resolution': [1],
        'kind': ['cerebellar', 'nifti'],
    },
}


def list_atlases(name_only=True):
    """
    List all the available atlases.

    Parameters
    ----------
    name_only : bool
        If True (default), return a list of available atlas names. If False,
        return a dict of available atlas names including their valid
        parameters.
    Returns
    -------
    out : list(str) or dict
        A list or dict with all available atlases.
    """
    out = None
    if name_only is True:
        out = list(_available_atlases.keys())
    else:
        out = _available_atlases
    return out

# -----------------------------------------------------------------------------#
# Masks related
# -----------------------------------------------------------------------------#


def get_atlas_params(atlas_name):
    """
    Separate the atlas family name and the atlas specific further parameters.

    Parameters (required)
    ---------------------
    atlas_name : str
        Specify the atlas to be loaded by its name following the Junifer naming
        convention. Follow the principle
        <AtlasfamilynameAtlasxAdditionalparameterxAdditionalparameter>. E.g.
        the name for the Schaefer atlas (atlas family) with 1000 ROIs (atlas)
        and 7 yeo networks (additional parameter) and at a resolution of 1 mm
        (additional parametr) would be 'Schaefer1000x7x1'.
        Check valid options by calling list_atlases(name_only=False).

    Returns
    -------
    atlas_famname : str
        Name of the atlas family (e.g. 'Schaefer')
    atlas_params : dict
        Parameters further specifying the atlas family.
    """
    # get atlas family name
    atlas_famname = re.split(r'(\d+)', atlas_name.split('x')[0])[0]

    # atlas specific parameters
    if atlas_famname == 'Schaefer':
        # Get parameters from atlas_name (defaults are set in _retrieve_atlas)
        if len(atlas_name.split('x')) >= 1:
            atlas_params = {
                'n_rois': int(re.split(r'(\d+)', atlas_name.split('x')[0])[1])}
        if len(atlas_name.split('x')) == 3:
            resolution = int(atlas_name.split('x')[2])
            atlas_params['resolution'] = resolution
        if len(atlas_name.split('x')) == 2:
            yeo_network = int(atlas_name.split('x')[1])
            atlas_params['yeo_network'] = yeo_network

    elif atlas_famname == 'SUIT':
        # Get parameters from atlas_name (defaults are set in _retrieve_atlas)
        if len(atlas_name.split('x')) == 1:
            atlas_params = {}
        if len(atlas_name.split('x')) == 2:
            atlas_params = {
                'space': atlas_name.split('x')[1]}
        if len(atlas_name.split('x')) == 3:
            resolution = int(atlas_name.split('x')[2])
            atlas_params['resolution'] = resolution

    return atlas_famname, atlas_params


def load_atlas(atlas_famname, out_dir='data/masks', **kwargs):
    """
    Loads a brain atlas (including a label file, description and kind of atlas)
    for parcellation. If the atlas is not yet in the out_dir it retrieves the
    atlas first and saves it in out_dir.

    Parameters (required)
    ---------------------
    atlas_famname : str
        Specify by name of atlas family, e.g. 'Schaefer'.
        Check valid options by calling list_atlases(name_only=True).
    out_dir: path
        Path to where to store the retrieved atlas file.
        Default (relative to base of project structure): data/masks

    Parameters (optional, atlas dependent)
    --------------------------------------
    Use to specify atlas specific keyword arguments. Check valid options by
    calling list_atlases(name_only=False).

    Schaefer :
        n_rois (required) : int
            Granularity of atlas to be used. Valid values: between 100 and 1000
            (included) in steps of 100.
        yeo_network (optional) : int
            Number of yeo networks to use. Valid values: 7, 17. Defaults to 7.
        resolution (optional) : int
            Resolution of atlas nifti to use. Valid values: 1, 2. Defaults to 1.

    SUIT :
        space (optional) : str
            Space of atlas can be either 'MNI' or 'SUIT' (for more information
            see http://www.diedrichsenlab.org/imaging/suit.htm). Defaults to
            'MNI'.

    Returns
    -------
    atlas_img : niimg-like object (depending on atlas)
        Loaded atlas image.
    atlas_labels : List of str
        Atlas labels.
    atlas : atlas object
        Atlas object as retrieved with _retrieve_atlas() for further operations.
    """

    # retrieve atlases by passing arguments on to _retrieve_atlas()
    atlas = _retrieve_atlas(
        atlas_famname, out_dir, **kwargs)

    # atlas retrieved from
    logger.info(
        f'Load atlas {PurePath(Path(atlas["maps"])).name} from directory '
        f'{Path(atlas["maps"]).parent}.')

    # load atlas img
    # load nifti
    atlas_img = image.load_img(atlas["maps"])

    atlas_labels = atlas['labels']

    return atlas_img, atlas_labels, atlas


def _retrieve_atlas(atlas_famname, out_dir='data/masks', **kwargs):
    """
    Retrieves a brain atlas object either from nilearn or a specified online
    source. Only returns one atlas per call. Call function multiple times for
    different parameter specifications. Only retrieves atlas if it is not yet in
    out_dir.

    Parameters (required)
    ---------------------
    atlas_famname : str
        Specify by name of atlas family, e.g. 'Schaefer'.
        Check valid options by calling list_atlases(name_only=True).
    out_dir: path
        Path to where to store the retrieved atlas file.
        Default (relative to base of project structure): data/masks

    Parameters (optional, atlas dependent)
    --------------------------------------
    Use to specify atlas specific keyword arguments. Check valid options by
    calling list_atlases(name_only=False).

    Schaefer :
        n_rois (required) : int
            Granularity of atlas to be used. Valid values: between 100 and 1000
            (included) in steps of 100.
        yeo_network (optional) : int
            Number of yeo networks to use. Valid values: 7, 17. Defaults to 7.
        resolution (optional) : int
            Resolution of atlas nifti to use. Valid values: 1, 2. Defaults to 1.

    SUIT :
        space (optional) : str
            Space of atlas can be either 'MNI' or 'SUIT' (for more information
            see http://www.diedrichsenlab.org/imaging/suit.htm). Defaults to
            'MNI'.

    Returns
    -------
    atlas : dict
        Dictionary with the following keys:
        maps : str - path to atlas file
        labels : labels of atlas ROIs
        description : Only if retrieved from nilearn. Background info on atlas
        kind : str - indicating type of atlas (volumetric/surface) by file name
               extension (e.g. nifti)
    """

    _valid_atlases = list_atlases(name_only=True)

    out_dir = Path(out_dir)
    out_dir_atl = out_dir / atlas_famname
    out_dir_atl.mkdir(exist_ok=True, parents=True)

    if atlas_famname in _valid_atlases:
        logger.info(f"{atlas_famname} atlas was selected.")

        # retrieval details per atlas
        if atlas_famname == 'Schaefer':
            # valid atlas parameters
            _valid_nrois = list_atlases(
                name_only=False)[atlas_famname]['valid_nrois']
            _valid_resolution = list_atlases(
                name_only=False)[atlas_famname]['valid_resolution']
            _valid_yeonetworks = list_atlases(
                name_only=False)[atlas_famname]['valid_yeonetworks']
            kind = list_atlases(
                name_only=False)[atlas_famname]['kind'][1]

            # set atlas defaults for **kwargs parameters
            resolution = kwargs.get('resolution', 1)
            yeo_network = kwargs.get('yeo_network', 7)
            logger.info(
                'The atlas parameters "resolution" and "yeo_network" were set '
                f'to {resolution} mm and {yeo_network} networks, respectively.')

            # check missing atlas parameter specifications in **kwargs
            if 'n_rois' not in kwargs:
                raise_error(
                    'The specification of the "n_rois" parameter (atlas '
                    'granualrity) is a required keyword parameter for the '
                    'schaefer atlas.')
            else:
                n_rois = kwargs.get('n_rois')

            # check validity of atlas parameters
            if n_rois not in _valid_nrois:
                raise_error(
                    f'The parameter "n_rois" for the {atlas_famname} atlas '
                    f'needs to be within {_valid_nrois}. n_rois={n_rois}'
                    ' was specified instead.')
            if resolution not in _valid_resolution:
                raise_error(
                    f'The parameter "resolution" for the {atlas_famname} atlas '
                    f'needs to be within {_valid_resolution}. resolution = '
                    f'{resolution} was specified instead.')
            if yeo_network not in _valid_yeonetworks:
                raise_error(
                    f'The parameter "yeo_network" for the {atlas_famname} atlas'
                    f' needs to be within {_valid_yeonetworks}. yeo_network = '
                    f'{yeo_network} was specified instead.')

            # define file names
            atlas_fname = out_dir_atl / (
                f'Schaefer2018_{n_rois}Parcels_{yeo_network}Networks_order_'
                f'FSLMNI152_{resolution}mm.nii.gz')
            atlas_tname = out_dir_atl / (
                f'Schaefer2018_{n_rois}Parcels_{yeo_network}Networks_order.txt')

            # check existance of atlas
            if atlas_fname.exists() and atlas_tname.exists():
                logger.info(
                    f'The atlas {PurePath(atlas_fname).name} and '
                    f'its metadata file exist in {out_dir_atl}.')
                labels = [
                    '_'.join(x.split('_')[1:])
                    for x in pd.read_csv(
                        atlas_tname, sep='\t', header=None).iloc[:, 1].to_list()
                ]
                description = _get_dataset_descr('schaefer_2018')
                atlas = {
                    'maps': atlas_fname.as_posix(),
                    'labels': labels,
                    'description': description,
                    'kind': kind,
                }

            # fetch atlas from nilearn
            else:
                atlas = (
                    datasets.fetch_atlas_schaefer_2018(
                        n_rois=n_rois,
                        yeo_networks=yeo_network,
                        resolution_mm=resolution,
                        data_dir=out_dir.as_posix()))
                # move atlas from nilearn directory to atlas family directory
                shutil.copy2(atlas["maps"], atlas_fname)  # nifti
                shutil.copy2(
                    PurePath(atlas["maps"]).parent / PurePath(atlas_tname).name,
                    atlas_tname)  # metadata
                # delete nilearn created directory
                shutil.rmtree(PurePath(atlas["maps"]).parent)
                # change directory entry in atlas object
                atlas['maps'] = atlas_fname.as_posix()
                atlas['kind'] = kind
                atlas['labels'] = [
                    '_'.join(x.split('_')[1:]) for x in atlas.labels.astype('U')
                ]
                logger.info(
                    f'The atlas {PurePath(atlas_fname).name} and '
                    'its metadata file DID NOT exist in '
                    f'{out_dir_atl} and were retrieved from '
                    'nilearn.')

        # elif atlas_famname == 'Tian':
        #     # valid atlas parameters
        #     _valid_nrois = list_atlases(
        #         name_only=False)[atlas_famname]['valid_nrois']
        #     _valid_resolution = list_atlases(
        #         name_only=False)[atlas_famname]['valid_resolution']
        #     _valid_yeonetworks = list_atlases(
        #         name_only=False)[atlas_famname]['valid_yeonetworks']
        #     kind = list_atlases(
        #         name_only=False)[atlas_famname]['kind'][1]

        elif atlas_famname == 'SUIT':
            # valid atlas parameters
            _valid_resolution = list_atlases(
                name_only=False)[atlas_famname]['valid_resolution']
            _valid_space = list_atlases(
                name_only=False)[atlas_famname]['valid_space']
            kind = list_atlases(
                name_only=False)[atlas_famname]['kind'][1]
            description = (
                'Probabilistic atlas for cerebellar lobules. Known as '
                'SUIT atlas or Diedrichsen 2009 (Diedrichsen, J., Balsters, '
                'J. H., Flavell, J., Cussans, E., & Ramnani, N. (2009). A '
                'probabilistic atlas of the human cerebellum. Neuroimage.46(1),'
                ' 39-46.)). For more information see: http://www.diedrichsenlab'
                '.org/imaging/suit.htm. Atlas retrieved from https://github.com'
                '/DiedrichsenLab/cerebellar_atlases/tree/master/Diedrichsen_'
                '2009.')

            # set atlas defaults for **kwargs parameters
            resolution = kwargs.get('resolution', 1)
            space = kwargs.get('space', 'MNI')
            logger.info(
                'The atlas parameters "resolution" and "space" were set to '
                f'{resolution} mm and {space} space, respectively.')

            # check validity of atlas parameters
            if resolution not in _valid_resolution:
                raise_error(
                    f'The parameter "resolution" for the {atlas_famname} atlas '
                    f'needs to be within {_valid_resolution}. resolution = '
                    f'{resolution} was specified instead.')
            if space not in _valid_space:
                raise_error(
                    f'The parameter "space" for the {atlas_famname} atlas '
                    f'needs to be within {_valid_space}. space = '
                    f'{space} was specified instead.')

            # define file names
            atlas_fname = out_dir_atl / (
                f'SUIT_{space}Space_{resolution}mm.nii')
            atlas_tname = out_dir_atl / (
                f'SUIT_{space}Space_{resolution}mm.tsv')

            # check existance of atlas
            if atlas_fname.exists() and atlas_tname.exists():
                logger.info(
                    f'The atlas {PurePath(atlas_fname).name} and '
                    f'its label file exist in {out_dir_atl}.')
                atlas = {
                    'maps': atlas_fname.as_posix(),
                    'labels': pd.read_csv(
                        atlas_tname,
                        sep='\t', usecols=['name'])['name'].to_list(),
                    'description': description,
                    'kind': kind,
                }

            # fetch atlas from github
            else:
                url_basis = (
                    'https://github.com/DiedrichsenLab/cerebellar_atlases/raw'
                    '/master/Diedrichsen_2009/')
                url_MNI = url_basis + 'atl-Anatom_space-MNI_dseg.nii'
                url_SUIT = url_basis + 'atl-Anatom_space-SUIT_dseg.nii'
                url_labels = url_basis + 'atl-Anatom.tsv'
                # get labels
                labels_download = requests.get(url_labels)
                labels = pd.read_csv(
                    io.StringIO(labels_download.content.decode("utf-8")),
                    sep='\t', usecols=['name'])['name'].to_list()
                with open(atlas_tname, 'wb') as f:
                    f.write(labels_download.content)  # save df not down.content
                # download atlas and save
                if space == 'MNI':
                    atlas_download = requests.get(url_MNI)
                    with open(atlas_fname, 'wb') as f:
                        f.write(atlas_download.content)
                elif space == 'SUIT':
                    atlas_download = requests.get(url_SUIT)
                    with open(atlas_fname, 'wb') as f:
                        f.write(atlas_download.content)
                # build atlas dictionary to be consistent with nilearn
                atlas = {
                    'maps': atlas_fname.as_posix(),
                    'labels': labels,
                    'description': description,
                    'kind': kind,
                }
                logger.info(
                    f'The atlas {PurePath(atlas_fname).name} and '
                    'its metadata file DID NOT exist in '
                    f'{out_dir_atl} and were retrieved from '
                    f'{url_basis}.')

    else:
        raise_error(
            f"The provided atlas name {atlas_famname} cannot be retrieved. "
            f"Please choose one of these options: {_valid_atlases}")

    return atlas


# -----------------------------------------------------------------------------#
# Structural features related
# -----------------------------------------------------------------------------#

def get_vbm(CAT_REPO_URL, target_dir, dataset_name, subid, session):
    """
    Retrieves the preprocessed VBM data from the CAT DataLad dataset.

    Parameters
    ----------
    CAT_REPO_URL : str
        URL of CAT preprocessed DataLad dataset.
    target_dir : str
        General directory to install DataLad dataset to. The DataLad dataset
        will be created as a subdirectory with the specified dataset_name in
        this directory.
    dataset_name : str
        Name of the dataset. Used to create a subdirectory in target_dir to
        install the datalad dataset to.
    subid : str
        Subject ID as used in the CAT preprocessed DataLad dataset.
    session: str
        Session ID (e.g. 'ses-2') as used in the CAT preprocessed DataLad
        dataset.

    Returns
    -------
    vbm_nifti : Niimg like object
        VBM nifti for a subject and session as preprocessed by CAT and stored
        in specified DataLad dataset.
    """

    # definitions
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)
    install_dir = target_dir / dataset_name  # existance established by datalad
    get_dir = install_dir / 'm0wp1'  # defined by dataset structure

    # clone VBM datalad dataset
    dl.install(path=install_dir, source=CAT_REPO_URL)
    logger.info(
        f'VBM DataLad dataset was installed from {CAT_REPO_URL} to '
        f'{install_dir}')

    # create nifti-image path
    nifti_fname = (
        get_dir / f'm0wp1{subid}_{session}_T1w.nii.gz')

    # check existance of cloned symbolic links
    if not nifti_fname.is_symlink():
        raise_error(
            f'VBM image file name "m0wp1{subid}_{session}_T1w.nii.gz" does not '
            f'exist in: {nifti_fname.as_posix()}')

    # get data (of predefined sbj, ses)
    dl.get(path=nifti_fname, dataset=install_dir)
    logger.info(
        'Symlinks from dataset installation existed. Got VBM nifti for subject '
        f'{subid} and session {session}.')

    # load nifti
    vbm_nifti = image.load_img(nifti_fname.as_posix())
    logger.info('VBM nifti was loaded.')

    return vbm_nifti


def get_gmd(atlas_nifti, vbm_nifti, aggregation=None, limits=None):
    """
    Constructs a masker based on the input atlas_nifti, applies resampling of
    the atlas if necessary and applies the masker to
    the vbm_nifti to extract (and return) measures of region-wise gray matter
    density (GMD). So far the aggregtaion methods "winsorized mean", "mean" and
    "std" are supported.

    Parameters
    ----------
    atlas_nifti : niimg-like object
        Nifti of atlas to use for parcellation.
    vbm_nifti: niimg-like object
        Nifti of voxel based morphometry as e.g. outputted by CAT.
    aggregation: list
        List with strings of aggregation methods to apply. Defaults to
        aggregation = ['winsorized_mean', 'mean', 'std'].
    limits: array
        Array with lower and upper limit for the calculation of the winsorized
        mean. Only needed when 'winsorized_mean' was specified
        in aggregation. If wasn't specified defaults to [0.1, 0.1].

    Returns
    -------
    gmd_aggregated : dict
        Dictionary with keys being each of the chosen aggregation methods
        and values the corresponding array with the calculated GMD based on the
        provided atlas. The array therefore as the shape of the chosen number
        of ROIs (granularity).
    agg_func_params: dict
        Dictionary with parameters used for the aggregation function. Keys:
        respective aggregation function, values: dict with responding
        parameters
    """

    # defaults (validity is checked in _get_funcbyname())
    if aggregation is None:  # Don't put mutables as defaults, use None instead
        aggregation = ['winsorized_mean', 'mean', 'std']
    if limits is None:
        limits = [0.1, 0.1]

    # aggregation function parameters (validity is checked in _get_funcbyname())
    agg_func_params = {'winsorized_mean': {'limits': limits}}

    # definitions
    # sort rois to be related to the order of i_roi (and get rid of 0 entry)
    rois = sorted(np.unique(image.get_data(atlas_nifti)))[1:]  # roi numbering
    n_rois = len(rois)  # granularity
    gmd_aggregated = {x: np.ones(shape=(n_rois)) * np.nan for x in aggregation}

    # resample atlas if needed
    if not atlas_nifti.shape == vbm_nifti.shape:
        atlas_nifti = image.resample_to_img(
            atlas_nifti, vbm_nifti, interpolation='nearest')
        logger.info('Atlas nifti was resampled to resolution of VBM nifti.')

    # make masker and apply
    for i_roi, roi in enumerate(rois):
        mask = image.math_img(f'img=={roi}', img=atlas_nifti)
        gmd = masking.apply_mask(imgs=vbm_nifti, mask_img=mask)  # gmd per roi
        logger.info(f'Mask applied for roi {roi}.')
        # aggregate (for all aggregation options in list)
        for agg_name in aggregation:
            logger.info(f'Aggregate GMD in roi {roi} using {agg_name}.')
            agg_func = _get_funcbyname(
                agg_name, agg_func_params.get(agg_name, None))
            gmd_aggregated[agg_name][i_roi] = agg_func(gmd)
    logger.info(f'{aggregation} was computed for all {n_rois} ROIs.\n')

    return gmd_aggregated, agg_func_params


# -----------------------------------------------------------------------------#
# Surface features related
# -----------------------------------------------------------------------------#


def ukbb_get_searched_categories_description(html):
    """
    Given an html link to the result of a UKBB showcase category search
    returns a dictionary with the keys being the Category ID and values the
    corresponding description.

    Parameters
    ----------
    html : str
        url of result of a UKBB showcase category search.

    Returns
    -------

    categories_description : dict
        dictionary with keys being the category ID and values the corresponding
        description.
    """
    categories_description = pd.read_html(html)[2].drop(columns='Items')
    categories_description = dict(zip(
      categories_description['Category ID'],
      categories_description['Description']))
    categories_description = {
        k: v.replace(" ", "_") for k, v in categories_description.items()}

    return categories_description


def ukbb_get_category_datafields(category, session=None, addSbjID=True):
    """
    Returns a list of all datafield IDs contained in the specified category. 
    Adds the specified session to match the datafield naming in the columns of 
    downloaded UKBB conent.

    Parameters
    ----------

    category : int
        Category ID as specified in the UKBB showcase.
    session : string, default: 'ses-2'
        Session ID. Only the imaging session 'ses-2' and 'ses-3' are currently
        supported.
    addSbjID : Boolean, default: True
        Set to false if Subject ID shall not be added as column.

    Returns
    -------

    datafields : list
        List containing all datafields of the specified category in the format
        'datafieldID_sessionID'.
    description : list
        List containing the labeled name of the datafield.
    """
    if session is None:
        ses_label = '-2.0'
    elif session == 'ses-2':
        ses_label = '-2.0'
    elif session == 'ses-3':
        ses_label = 'ses-3'
    else:
        raise_error(
            "Invalid session specification. Specify session='ses-2' or "
            "session='ses-3.")
    category_df = pd.read_html(
      "http://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id="+str(category))[0]
    datafields = [str(x) + ses_label for x in list(category_df['Field ID'])]
    description = list(category_df['Description'])
    if addSbjID is True:
        datafields[len(datafields):] = ["eid"]
        description[len(description):] = ["SubjectID"]

    return datafields, description


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#

# generic way of applying any function
def _get_funcbyname(name, func_params):
    """
    Helper function to generically apply any function. Here used to apply
    different aggregation functions for extraction of gray matter density (GMD).

    Parameters
    ----------
    name : str
        Name to identify the function. Currently supported names and
        corresponding functions are:
        'winsorized_mean' -> scipy.stats.mstats.winsorize
        'mean' -> np.mean
        'std' -> np.std

    func_params : dict
        Dictionary containing functions that need further parameter
        specifications. Keys are the function and values are dictionaries
        with the parameter specifications.
        E.g. 'winsorized_mean': func_params = {'limits': [0.1, 0.1]}

    Returns
    -------
    respective function with inputted (partial) parameters.
    """

    # check validity of names
    _valid_func_names = {'winsorized_mean', 'mean', 'std'}

    # apply functions
    if name == 'winsorized_mean':
        # check validity of func_params
        limits = func_params.get('limits')
        if all((lim >= 0.0 and lim <= 1) for lim in limits):
            logger.info(f'Limits for winsorized mean are set to {limits}.')
        else:
            raise_error(
                'Limits for the winsorized mean must be between 0 and 1.')
        # partially interpret func_params
        return partial(winsorized_mean, **func_params)
    if name == 'mean':
        return np.mean  # No func_params
    if name == 'std':
        return np.std

    else:
        raise_error(f'Function {name} unknown. Please provide any of '
                    f'{_valid_func_names}')


def winsorized_mean(data, axis=None, **win_params):
    """
    Helper function to chain winsorization and mean to compute winsorized
    mean.

    Parameters
    ----------
    data : array
        Data to calculate winsorized mean on.
    win_params : dict
        Dictionary containing the keyword arguments for the winsorize function.
        E.g. {'limits': [0.1, 0.1]}

    Returns
    -------
    Winsorized mean of the inputted data with the winsorize settings applied
    as specified in win_params.
    """

    win_dat = winsorize(data, axis=axis, **win_params)
    win_mean = win_dat.mean(axis=axis)

    return win_mean
