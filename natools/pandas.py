import pandas as pd
import json
import sys


def print_full_dataframe(df, file_path=None):
    """
    :param df: pandas Dataframe to print
    :param file_path: path where to save the Dataframe printing (optional)
    :return: None, but prints and save stuff to disk
    """
    pd.set_option('display.max_rows', len(df))

    print(df)

    if file_path is not None:
        orig_stdout = sys.stdout
        with open(file_path, 'w') as f:
            sys.stdout = f
            print(df)
        sys.stdout = orig_stdout
    pd.reset_option('display.max_rows')


def fill_in_dataframe_set(df, index, **kwargs):
    """
    :param df: pandas Dataframe
    :param index: index of the row to add
    :param kwargs: columns values
    :return: None (modifications of a pandas Dataframe are inplace)
    """
    for key, value in kwargs.items():
        if value:
            df.loc[index, key] = value


def fill_in_dataframe_increment(df, index, **kwargs):
    """
    :param df: pandas Dataframe
    :param index: index of the row to add
    :param kwargs: columns values
    :return: None (modifications of a pandas Dataframe are inplace)
    """
    for key, value in kwargs.items():
        if value:
            df.loc[index, key] += value


def add_constant_column(df, column_name, value):
    """
    :param df: pandas Dataframe
    :param column_name: new column's name (string)
    :param value: constant value of the new column
    :return: None (modifications of a pandas Dataframe are inplace)
    """
    df[column_name] = pd.Series([value] * df.shape[0], index=df.index)


def add_apply_column(df, column_name, function):
    """
    :param df: pandas Dataframe
    :param column_name: new column's name (string)
    :param function: function to apply (can be lambda or pre-defined function) on each row
    :return: None (modifications of a pandas Dataframe are inplace)
    
    Also valid to just *modify* a column.
    """
    df[column_name] = df.apply(function, axis=1)


def reverse_dataframe(df):
    return df.reindex(index=df.index[::-1]).reset_index(drop=True)


def convert_df_to_dict(df):
    if df is not None:
        records = json.loads(df.T.to_json()).values()
        return records
    else:
        return None

if __name__ == "__main__":
    pass
