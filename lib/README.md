# Library for the Confound Continuum Paper

## Installing

Go to the directory where the `setup.py` file is and execute (in the right virtual environment)

```
python setup.py develop
```

## Tools:

### Logging

First, set up logging (to stdout)

```Python
from confoundcontinuum.logging import configure_logging, log_versions


configure_logging()
log_versions()
```

Then, you can just use the available methods:

```Python
from confoundcontinuum.logging import logger

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning')
```

If you want to raise an error:

```Python
from confoundcontinuum.logging import raise_error

raise_error('This is an error message')
```

### Storing dataframes in sqlite files

Check `confoundcontinuum.io`:

```
Signature:
save_features(
    df,
    uri,
    kind,
    atlas_name,
    agg_function=None,
    if_exist='append',
)
Docstring:
Save features to a SQL Database

Parameters
----------
df : pandas.DataFrame
    The Pandas DataFrame to save. Must have the index set.
uri : str
    The connection URI.
    Easy options: 
        'sqlite://' for an in memory sqlite database
        'sqlite:///<path_to_file>' to save in a file
    
    Check https://docs.sqlalchemy.org/en/14/core/engines.html for more
    options
kind : str
    kind of features
altas_name : str
    the name of the atlas
agg_function : str
    The aggregation function used (defaults to None)
if_exist : str
    How to behave if the table already exists. Options are:
    'replace': Drop the table before inserting new values.
    'append': Insert new values to the existing table (default).
File:      ~/dev/projects/ConfoundContinuum/lib/confoundcontinuum/io.py
Type:      function
```

```
Signature: read_features(uri, kind, atlas_name, index_col, agg_function=None)
Docstring:
Read features from a SQL Database

Parameters
----------
uri : str
    The connection URI.
    Easy options: 
        'sqlite://' for an in memory sqlite database
        'sqlite:///<path_to_file>' to save in a file
    
    Check https://docs.sqlalchemy.org/en/14/core/engines.html for more
    options
kind : str
    kind of features
altas_name : str
    the name of the atlas
index_col : list(str)
    The columns to be used as index
agg_function : str
    The aggregation function used (defaults to None)

Returns
-------
df : pandas.DataFrame
    The DataFrame with the features
File:      ~/dev/projects/ConfoundContinuum/lib/confoundcontinuum/io.py
Type:      function
```

