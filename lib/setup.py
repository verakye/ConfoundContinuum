#! /usr/bin/env python
#

import os
from pathlib import Path

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup
from setuptools import find_packages

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(Path('confoundcontinuum') / '__init__.py', 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

_ROOT = Path(__file__).parent.resolve()

descr = """Motor Prediction project."""

DISTNAME = 'confoundcontinuum'
DESCRIPTION = descr
MAINTAINER = '@fraimondo'
MAINTAINER_EMAIL = 'f.raimondo@fz-juelich.de'
URL = 'https://github.com/verakye/ConfoundContinuum.git'
LICENSE = 'Copyright'
DOWNLOAD_URL = 'https://github.com/verakye/ConfoundContinuum.git'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=find_packages(
              exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
          scripts=[])
