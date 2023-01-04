import os
import pandas as pd
import glob
import json
from typing import Dict, List
from scipy.fft import fft, fftfreq
import numpy as np
from matplotlib import pyplot as plt

# Loading csv dataset

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
        # for key in data_off.keys():
        #     add_intensity(data_off[key])
        # for key in data_on.keys():
        #     add_intensity(data_on[key])
        return data_on, data_off
    else:
        raise Exception("cannot process 6 classes")

# feature generation

def add_intensity(df: pd.DataFrame):
    """
    add Intensity as 2nd norm of 3D vector
    """
    df["Intensity"] = (df["X_UnCal"] ** 2 + df["Y_UnCal"] ** 2 + df["Z_UnCal"] ** 2) ** 0.5

def add_intensity_bias(df: pd.DataFrame):
    """
    add Intensity bias as 2nd norm of 3D vector
    """
    df["Bias_Intensity"] = (df["X_Bias"] ** 2 + df["Y_Bias"] ** 2 + df["Z_Bias"] ** 2) ** 0.5

def add_intensity_to_dataset(data: Dict[str, pd.DataFrame]) -> None:
    """
    In-place adds intensity to dataset
    """
    for key in data.keys():
            add_intensity(data[key])
            add_intensity_bias(data[key])

# usual statistical features
def statistical_features_flat(df: pd.DataFrame, test_case_name: str, test_case_class: int) -> dict:
    

    d = dict() 

    # add test_case_name and label
    d['Name'] = test_case_name
    d['Label'] = test_case_class

    statistics = df[['X_UnCal', 'Y_UnCal', 'Z_UnCal', 'Intensity']].describe()
    for col_name in statistics.columns:
        for row_name, value in statistics.iterrows():

            # add one feature "count"
            if col_name == "X_UnCal" and row_name == "count":
                d["count"] = value[col_name]

            # exclude other counts, since they are the same
            if row_name != "count":
                d[f"{col_name}_{row_name}"] = value[col_name]
            

    # add features that do not change
    for feature in ['X_Bias', 'Y_Bias', 'Z_Bias', 'Bias_Intensity', 'Accuracy']:
        d[feature] = df[feature].iloc[0]

    
    return d


def statistical_dataset(data: Dict[str, pd.DataFrame], label: str) -> pd.DataFrame:
    return pd.DataFrame([statistical_features_flat(df=df, test_case_name=name, test_case_class=label)for name, df in data.items()])



def spectral_image(df: pd.DataFrame, column = 'Intensity', duration = 60):
    N = len(df)
    sampling_rate = N // duration
    if sampling_rate == 0:
        print(N)
        print(duration)
        print(df)
    T = 1 / sampling_rate

    yf = fft(df[column].to_numpy())

    # just right side
    yff = 2.0/N * np.abs(yf[0:N//2])
    xf = fftfreq(N, T)[:N//2]

    bins = np.array(range(round(xf[-1])))
    inds = np.digitize(xf, bins)
    return pd.DataFrame({"bins": inds, "fourier": yff}).groupby("bins").sum()



def specter_to_data_row(name, label, df, vec_len, column = 'Intensity', duration = 60,): 
    specter_intensity = spectral_image(df, column, duration)['fourier'].to_numpy()[:vec_len]
    padded = np.zeros((vec_len))
    padded[:specter_intensity.shape[0]] = specter_intensity
    data_row = [name, label] + list(padded)
    return data_row


def spectral_dataset(data: Dict[str, pd.DataFrame], label: int, vec_len: int = 50, experiment_duration = 60, column = 'Intensity') -> pd.DataFrame:
    """
    Creates a pandas Dataframe with test cases in rows with first vec_len frequencies. It begins with name of text_case and label
    """
    columns = ["Name", "Label"] + list(range(1,vec_len+1))

    return pd.DataFrame([specter_to_data_row(name, label=label, df=df, vec_len=vec_len, column=column, duration=experiment_duration) for name, df in data.items()], columns=columns)



# joining, ordering datasets
def join_and_order_dataset(dfs: List[pd.DataFrame], sort_by = ['Label', 'Name']):
    dataset = pd.concat(dfs, ignore_index=True)
    return dataset.sort_values(by=sort_by, ignore_index=True)



# save dataset

def save_dataset(dataset: pd.DataFrame, dir_path: str, dataset_name: str, metadata: dict) -> None:
    # save dataset
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_name = f"{metadata['version']}-{metadata['stages']}_stages-{dataset_name}"
    dataset.to_csv(f"{dir_path}/{file_name}.csv",index=True)
    # Writing metadata
    with open(f"{dir_path}/{file_name}.json", "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
