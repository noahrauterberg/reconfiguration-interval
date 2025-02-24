import pandas as pd
import os
import typing


def load_file(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the given path.

    :param path: The path to the CSV file.
    :return: The CSV file as a pandas DataFrame.
    """
    return pd.read_csv(path)


def load_dir(path: str) -> typing.Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the given directory.

    :param path: The path to the directory.
    :return: A list of pandas DataFrames.
    """
    ret = {}
    for f in os.listdir(path):
        if f == "545.csv":
            print("FOUND")
        if not f.endswith(".csv"):
            continue
        df = load_file(f)
        time = f.removesuffix(".csv")
        ret.time = df

    return ret
