import os
import pandas as pd
import glob
import json
from typing import Dict, List
from scipy.fft import fft, fftfreq
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

# loading raw data

def remove_from_string(test_string):
    """
    Reads test names and locations from filenames
    """
    # remove comments
    remove_list = ["resorant", "Trafic", "Traffic", "With", "Truffic", "Eith", "DifferentOrentationPousesInRuns", "Orient"]
    for sub in remove_list:
        test_string = test_string.replace(sub, '')
    # remove numbers
    test_string = "".join(list(filter(lambda x: x.isalpha(), test_string)))
    return test_string

location_mapping = {
    "NadasKitchen": "Kitchen1",
    "MihasLivingRoom": "LivingRoom1",
    "milenkosRoomBed": "DormRoom1",
    "PiXmilenkosRoomTable": "DormRoom2",
    "spelaMilenkosRoomTable": "DormRoom3",
    "milenkoPark": "Park",
    "LATERNA": "Restaurant",
    "NadasLivingRoom": "LivingRoom2",
    "milenkosRoomNearDevices": "DormRoom4",
    "milenkosRoom": "DormRoom5",
    "milenkosRoomTable": "DormRoom6",
    "NadasBathroom": "Bathroom",
    "spelaMilenkosKitchenTable": "Kitchen2",
    "milenkosKitchenNotNearDevices": "Kitchen3",
    "PiXmilenkosRoomRotationsTable": "DormRoom7"
}

rotations_mapping = defaultdict(bool)
with_rotations = ["DormRoom2", "DormRoom7", "Kitchen1", "LivingRoom2", "DormRoom5", "DormRoom3",  "Kitchen2"]
for loc in with_rotations:
    rotations_mapping[loc] = True
device_mapping =  {
        "4aaf95a621ccf092": "RedmiNote8PRO",
        "029a77f196804217": "SamsungGalaxyA51",
        "03575768cc23b2df": "GooglePixel6",
        "e08d976ac75c011e": "SamsungGalaxyS6"
    }

def reverse_device_mapping(device_names: list):
    return [k for k,v in device_mapping.items() if v in device_names]

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
        return data_on, data_off
    else:
        raise Exception("cannot process 6 classes")

# feature generation

def standardize(df: pd.DataFrame, columns = ['Intensity']) -> None:
    """
    standardize data in columns
    """

    df[columns] = (df[columns]-df[columns].mean())/df[columns].std()


def standardize_dataset(data: Dict[str, pd.DataFrame], columns = ['Intensity']) -> None:
    """
    standardize entire dataset
    """
    for key in data.keys():
        standardize(data[key], columns)

def normalize(df: pd.DataFrame, columns = ['Intensity']) -> None:
    """
    normalize data in columns
    """

    df[columns] = (df[columns]-df[columns].mean())/(df[columns].max() - df[columns].max())


def normalize_dataset(data: Dict[str, pd.DataFrame], columns = ['Intensity']) -> None:
    """
    normalize entire dataset
    """
    for key in data.keys():
        normalize(data[key], columns)

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

    # add metadata
    d['name'] = test_case_name
    d['location'] = df['location'].iloc[0]
    d['device_id'] = df['device_id'].iloc[0]
    d['label'] = test_case_class
    

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
    location = df['location'].iloc[0]
    device_id = df['device_id'].iloc[0]
    specter_intensity = spectral_image(df, column, duration)['fourier'].to_numpy()[:vec_len]
    padded = np.zeros((vec_len))
    padded[:specter_intensity.shape[0]] = specter_intensity
    
    data_row = [name, location, device_id, label] + list(padded)
    return data_row


def spectral_dataset(data: Dict[str, pd.DataFrame], label: int, vec_len: int = 50, experiment_duration = 60, column = 'Intensity') -> pd.DataFrame:
    """
    Creates a pandas Dataframe with test cases in rows with first vec_len frequencies. It begins with name of text_case and label
    """
    columns = ["name", "location", "device_id", "label"] + list(range(1,vec_len+1))

    return pd.DataFrame([specter_to_data_row(name, label=label, df=df, vec_len=vec_len, column=column, duration=experiment_duration) for name, df in data.items()], columns=columns)



# joining, ordering datasets
def join_and_order_dataset(dfs: List[pd.DataFrame], sort_by = ['label', 'name']):
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
