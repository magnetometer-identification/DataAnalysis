import os
import pandas as pd
import glob
import json


def read_dataset(data_path, classes = 2):
    """
    returns two dicts, data_on and data_off, with intensity
    """
    if classes == 2:
        test_cases = [file.split("-")[2] for file in glob.glob(f"{data_path}/*stage_5*.csv")]
        for i,test_case in enumerate(test_cases):
            print(i,test_case)
        data_off = {test_case: pd.read_csv(glob.glob(f"{data_path}/*stage_5*{test_case}*")[0]) for test_case in test_cases}
        data_on = {test_case: pd.read_csv(glob.glob(f"{data_path}/*stage_6*{test_case}*")[0]) for test_case in test_cases}
        for key in data_off.keys():
            add_intensity(data_off[key])
        for key in data_on.keys():
            add_intensity(data_on[key])
        return data_on, data_off
    else:
        raise Exception("cannot process 6 classes")

def add_intensity(df: pd.DataFrame):
    """
    add Intensity as 2nd norm of 3D vector
    """
    df["Intensity"] = (df["X_UnCal"] ** 2 + df["Y_UnCal"] ** 2 + df["Z_UnCal"] ** 2) ** 0.5


