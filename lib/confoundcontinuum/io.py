import pandas as pd
# from pandas.core.base import NoNewAttributesMixin
# from pandas.io.sql import pandasSQL_builder
import numpy as np
from sqlalchemy import create_engine, inspect

from . logging import logger


def _get_existing_pk(con, table_name, index_col):
    pk_cols = ','.join(index_col)
    query = f'SELECT {pk_cols} FROM {table_name};'
    pk_indb = pd.read_sql(query, con=con)
    return pk_indb


def _split_incoming_data(df, pk_indb, index_col):
    incoming_pk = df.reset_index()[index_col]
    exists_mask = (
        incoming_pk[index_col]
        .apply(tuple, axis=1)
        .isin(pk_indb[index_col].apply(tuple, axis=1))
    )
    existing, new = df.loc[exists_mask.values], df.loc[~exists_mask.values]
    return existing, new


def _generate_update_statements(table, index_col, rows_to_update):
    from sqlalchemy import and_

    new_records = rows_to_update.to_dict(orient="records")
    pk_indb = rows_to_update.reset_index()[index_col]
    pk_cols = [table.c[key] for key in index_col]

    stmts = []
    for i, (_, keys) in enumerate(pk_indb.iterrows()):
        stmt = (
            table.update()
            .where(and_(col == keys[j]
                        for j, col in enumerate(pk_cols)))  # type: ignore
            .values(new_records[i])
        )
        stmts.append(stmt)
    return stmts


def _save_upsert(df, name, engine, upsert='ignore', if_exist='append'):
    if upsert not in ['delete', 'ignore']:
        raise ValueError('upsert must be either "delete" or "ignore"')

    # index_col = df.index.names
    with engine.begin() as con:
        if if_exist == 'replace':
            df.to_sql(name, con=con, if_exists='replace')
        else:
            df.to_sql(name, con=con, if_exists='append')
        # elif not inspect(engine).has_table(name):
        #     print('table does not exists')
        #     df.to_sql(name, con=con, if_exists='append')
        # else:
        #     pk_indb = _get_existing_pk(
        #         con, table_name=name, index_col=index_col)
        #     existing, new = _split_incoming_data(df, pk_indb, index_col)
        #     pandas_sql = pandasSQL_builder(con)
        #     pandas_sql.meta.reflect(only=[name])
        #     table = pandas_sql.get_table(name)
        #     update_stmts = NoNewAttributesMixin
        #     if upsert == 'delete':
        #         update_stmts = _generate_update_statements(
        #             table, index_col, existing)
        #         for stmt in update_stmts:
        #             con.execute(stmt)

        #     new.to_sql(name, con=con, if_exists='append')


def _validate_names(kind, atlas_name, agg_function):
    if '$' in kind:
        raise ValueError("Feature kind must not have the special character $")
    if '$' in atlas_name:
        raise ValueError("Atlas name must not have the special character $")
    if agg_function is not None and '$' in agg_function:
        raise ValueError(
            "Agg function mamemust not have the special character $")


def _to_table_name(kind, atlas_name, agg_function):
    _validate_names(kind, atlas_name, agg_function)
    table_name = f'{kind}${atlas_name}'
    if agg_function is not None:
        table_name = f'{table_name}${agg_function}'
    return table_name


def _from_table_name(table_name):
    splitted = table_name.split('$')
    kind = splitted[0]
    atlas_name = splitted[1]
    agg_function = splitted[2]
    return kind, atlas_name, agg_function


def list_features(uri):
    """List features from a SQL Database

    Parameters
    ----------
    uri : str
        The connection URI.
        Easy options:
            'sqlite://' for an in memory sqlite database
            'sqlite:///<path_to_file>' to save in a file

        Check https://docs.sqlalchemy.org/en/14/core/engines.html for more
        options

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the features list
    """
    logger.debug(f'Listing features from DB {uri}')
    engine = create_engine(uri, echo=False)
    features = {'kind': [], 'atlas_name': [], 'agg_function': []}
    for t_name in inspect(engine).get_table_names():
        t_k, t_a, t_f = _from_table_name(t_name)
        features['kind'].append(t_k)
        features['atlas_name'].append(t_a)
        features['agg_function'].append(t_f)
    return pd.DataFrame(features)


def read_features(uri, kind, atlas_name, index_col, agg_function=None):
    """Read features from a SQL Database

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
    """
    table_name = _to_table_name(kind, atlas_name, agg_function)
    logger.debug(f'Reading data from DB {uri} - table {table_name}')
    engine = create_engine(uri, echo=False)
    df = pd.read_sql(table_name, con=engine, index_col=index_col)
    return df


def save_features(df, uri, kind, atlas_name, agg_function=None,
                  if_exist='append'):
    """Save features to a SQL Database

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
    """
    table_name = _to_table_name(kind, atlas_name, agg_function)
    logger.debug(f'Saving data from DB {uri} - table {table_name}')
    engine = create_engine(uri, echo=False)
    _save_upsert(df, table_name, engine, upsert='delete')


def read_prs(fname):
    data = np.genfromtxt(fname=fname, delimiter="\t")
    subj_id = data[0].astype(int)
    prs = data[1]
    vfunc = np.vectorize(lambda x: f'sub-{x}')
    subj_id = vfunc(subj_id)
    df = pd.DataFrame({'SubjectID': subj_id, 'prs': prs})
    return df.set_index('SubjectID').dropna()


def read_pheno(fname):
    df = pd.read_csv(fname, sep=',')
    df['SubjectID'] = [f'sub-{x}' for x in df['eid']]  # type: ignore

    df.drop(columns={'eid'}, inplace=True)  # type: ignore

    return df.set_index('SubjectID')  # type: ignore


def read_apoe(fname):
    data = np.genfromtxt(fname=fname, delimiter="\t", dtype=str)
    subj_id = data[0, 1:].astype(int)
    vfunc = np.vectorize(lambda x: f'sub-{x}')
    subj_id = vfunc(subj_id)
    data_dict = {'SubjectID': subj_id}
    for col in data[1:]:
        col_name = col[0]
        col_data = col[1:]
        data_dict[col_name] = col_data

    df = pd.DataFrame(data_dict).set_index('SubjectID')
    df['APOE'] = df['rs429358'] + df['rs7412']
    return df
