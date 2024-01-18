import pandas as pd
from pandas.testing import assert_frame_equal
import tempfile
from sqlalchemy import create_engine
from confoundcontinuum.io import _save_upsert, save_features, read_features

df1 = pd.DataFrame({
    'pk1': [1, 2, 3, 4, 5],
    'pk2': ['a', 'b', 'c', 'd', 'e'],
    'col1': [11, 22, 33, 44, 55],
    'col2': [111, 222, 333, 444, 555]
}).set_index(['pk1', 'pk2'])

df2 = pd.DataFrame({
    'pk1': [2, 5, 6],
    'pk2': ['b', 'e', 'f'],
    'col1': [2222, 5555, 66],
    'col2': [22222, 55555, 666]
}).set_index(['pk1', 'pk2'])

df_update = pd.DataFrame({
    'pk1': [1, 2, 3, 4, 5, 6],
    'pk2': ['a', 'b', 'c', 'd', 'e', 'f'],
    'col1': [11, 2222, 33, 44, 5555, 66],
    'col2': [111, 22222, 333, 444, 55555, 666]
}).set_index(['pk1', 'pk2'])

df_ignore = pd.DataFrame({
    'pk1': [1, 2, 3, 4, 5, 6],
    'pk2': ['a', 'b', 'c', 'd', 'e', 'f'],
    'col1': [11, 22, 33, 44, 55, 66],
    'col2': [111, 222, 333, 444, 555, 666]
}).set_index(['pk1', 'pk2'])


index_col = ['pk1', 'pk2']
table_name = 'test'


def test_upsert_ignore():
    with tempfile.TemporaryDirectory() as _tmpdir:
        uri = f'sqlite:///{_tmpdir}/test.db'
        engine = create_engine(uri, echo=False)

        # Save to SQL
        df1.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace'
        )

        c_df1 = pd.read_sql(table_name, con=engine, index_col=index_col)
        assert_frame_equal(df1, c_df1)

        _save_upsert(df2, table_name, engine, upsert='ignore')

        c_dfignore = pd.read_sql(table_name, con=engine, index_col=index_col)
        assert_frame_equal(c_dfignore, df_ignore)


def test_upsert_delete():
    with tempfile.TemporaryDirectory() as _tmpdir:
        uri = f'sqlite:///{_tmpdir}/test.db'
        engine = create_engine(uri, echo=False)
        # Save to SQL
        df1.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace'
        )
        c_df1 = pd.read_sql(table_name, con=engine, index_col=index_col)
        assert_frame_equal(df1, c_df1)
        _save_upsert(df2, table_name, engine, upsert='delete')

        c_dfupdate = pd.read_sql(table_name, con=engine, index_col=index_col)
        assert_frame_equal(c_dfupdate, df_update)

        _save_upsert(df2, table_name, engine, upsert='delete')

        c_dfupdate = pd.read_sql(table_name, con=engine, index_col=index_col)
        assert_frame_equal(c_dfupdate, df_update)


def test_io_features():
    with tempfile.TemporaryDirectory() as _tmpdir:
        uri = f'sqlite:///{_tmpdir}/test.db'
        save_features(df1, uri, 'vbm', 'schaefer_2010_100', 'mean')

        c_df1 = read_features(uri, 'vbm', 'schaefer_2010_100',
                              index_col=index_col,
                              agg_function='mean')

        assert_frame_equal(df1, c_df1)

        save_features(df2, uri, 'vbm', 'schaefer_2010_100', 'mean')

        c_dfupdate = read_features(
            uri, 'vbm', 'schaefer_2010_100', index_col=index_col,
            agg_function='mean')
        assert_frame_equal(c_dfupdate, df_update)
