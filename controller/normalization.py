import pandas as pd
import numpy as np
import os


def show_log(type_display: str, value: int = None) -> None:
    """Set the maximum number of rows to be displayed in Pandas DataFrames.
    Parameters:
        type_display (str): Type of display to set. Rows or Columns
        value (int): Maximum number of rows to display. If None, the default setting is used.
    """
    if type_display.lower() == 'columns':
        pd.set_option('display.max_columns', value)
    elif type_display.lower() == 'rows':
        pd.set_option('display.max_rows', value)
    else:
        print("Type display not recognized. Use 'rows' or 'columns'.")


def import_list_csv(path: str) -> np.array:
    """Import all CSV files from a specified directory and return them as a NumPy array of DataFrames.
    Parameters:
        path (str): String path where the CSV files are located.

    Returns:
        np.array: Array of DataFrames with the data from the CSV files. 
    """

    if os.path.exists(path):
        list_files = [file for file in os.listdir(
            path) if file.endswith('.csv')]

        list_dataframe = {f"df_{file.replace('.csv', '')}": import_csv(os.path.join(path, file))
                          for file in list_files}
        return list_dataframe
    else:
        print(f"The path {path} does not exist.")
        return np.array([])


def import_csv(path) -> pd.DataFrame:
    """Import a file CSV from path and return a Pandas Dataframe 

    Parameters:
        path (str): String path where the CVS is located.

    Returns:
        pd.DataFrame: DataFrame with the data from the CSV.
    """
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error al importar el archivo CSV: {e}")
        return pd.DataFrame()


def standarize_titles(dataframe: pd.DataFrame, special_characters: np.array = []) -> pd.DataFrame:
    """reads the columns name from the dataframe and returns a standarized version replacing special characters with underscores  

    Parameters:
        dataframe (pd.DataFrame): DataFrame to standarize the columns names.
        special_characters (np.array): Array with special characters to be replaced. By defect is empty

    Returns:
        pd.DataFrame: DataFrame special characters with underscores in the columns names.
    """
    if len(special_characters) == 0:
        columns_target = ','.join(dataframe.columns).lower().split(',')
    else:
        columns_target = ','.join(dataframe.columns).lower()
        for special_character in special_characters:
            columns_target = columns_target.replace(special_character, '_')
        columns_target = columns_target.split(',')

    counter = 0
    columns_fixed = {}
    for column in columns_target:
        columns_fixed[dataframe.columns[counter]] = column
        counter += 1

    return dataframe.rename(columns=columns_fixed)


def replace_null_values(dataframe_Column: pd.Series, value_replace) -> pd.Series:
    """Replace null values in a DataFrame column with a specified value.

    Parameters:
        dataframe_Column (pd.Series): DataFrame column to replace null values.
        value_replace: Value to replace null values with.

    Returns:
        pd.Series: DataFrame column with null values replaced.
    """
    return dataframe_Column.where(~dataframe_Column.isnull(), value_replace)


def drop_duplicates(dataframe: pd.DataFrame, by: list = []) -> None:
    """Drop duplicate rows from a DataFrame and reset the index.    

    Parameters:
        dataframe (pd.DataFrame): DataFrame to drop duplicate rows from.
        by (list): List of columns to consider for identifying duplicates. If empty, all columns are considered.
    """

    print("Initial number of rows:", len(dataframe))
    print("Number of duplicate rows:", dataframe.duplicated().sum())
    if len(by) > 0:
        dataframe.drop_duplicates(subset=by, inplace=True)
    else:
        dataframe.drop_duplicates(inplace=True)
    print("Final number of rows:", len(dataframe))
    print("Number of duplicate rows:", dataframe.duplicated().sum())


def __merge_dataframes__(df_left: pd.DataFrame, df_right: pd.DataFrame, on: str, how: str = 'inner') -> pd.DataFrame:
    """Merge two DataFrames on specified columns.

    Parameters:
        df_left (pd.DataFrame): Left DataFrame to merge.
        df_right (pd.DataFrame): Right DataFrame to merge.
        on (str): Column name to merge on.
        how (str): Type of merge to perform. Default is 'inner'.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.merge(df_left, df_right, on=on, how=how)
    return merged_df


def merge_list_dataframe(main_dataframe: pd.DataFrame, array_dataframes: list, on: str) -> pd.DataFrame:
    """Merge a list of DataFrames into a single DataFrame.

    Parameters:
        dataframes (list): List of DataFrames to merge.
        array_dataframes (list): List of DataFrames to merge.
        on (str): Column name to merge on.
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_dataframe = main_dataframe
    for dataframe in array_dataframes:
        merged_dataframe = __merge_dataframes__(
            merged_dataframe, dataframe, on=on, how='left')
    return merged_dataframe


def adjust_data_time(column: pd.Series, format_time: str = '%Y-%m-%d') -> pd.Series:
    """Convert a DataFrame column to datetime format.

    Parameters:
        column (pd.Series): DataFrame column to convert.
        format_time (str): Format of the datetime. Default is '%Y-%m-%d'.

    Returns:
        pd.Series: DataFrame column in datetime format.
    """
    return pd.to_datetime(column, format=format_time)
