import os
import numpy as np
import pandas as pd
from os.path import join
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_image(file_name: str, image_path: str = join("TrainingDataset", "TrainingData", "Images")):
    file_name = file_name.split('.')[0]
    image_id  = [int(x) for x in file_name.split('_') if x.isdigit()][0]
    image     = mpimg.imread(join(image_path, f"{image_id}.png"))
    plt.imshow(image)
    plt.show()


def describe_scanpaths(asd_path: str, td_path: str) -> pd.DataFrame:
    file_names = os.listdir(asd_path) + os.listdir(td_path)
    file_names = [file_name for file_name in file_names if ".txt" in file_name]
    headers = ["file_name", "image_id", "is_asd", "min", "median", "max", "n_participants"]
    data = []
    for file_name in file_names:
        is_asd    = ("ASD" in file_name)
        image_id  = [int(x) for x in file_name.split('.')[0].split('_') if x.isdigit()][0]
        file_path = join((asd_path if is_asd else td_path), file_name)
        df        = pd.read_csv(file_path)
        n_participants  = (df["Idx"] == 0).sum()
        fixation_counts = df[np.roll((df["Idx"].values == 0), shift=-1)]["Idx"].values + 1
        data.append(
            [file_name, image_id, is_asd, fixation_counts.min(), np.median(fixation_counts), fixation_counts.max(), n_participants]
        )
    return pd.DataFrame(data=data, columns=headers)


def collect_scanpaths(file_names: List[str], asd_path: str, td_path: str) -> pd.DataFrame:
    # headers = ["participant_id", "image_id", "fixation_id", "x", "y", "duration", "file_name"]
    dfs    = []
    for file_name in file_names:
        is_asd    = ("ASD" in file_name)
        image_id  = [int(x) for x in file_name.split('.')[0].split('_') if x.isdigit()][0]
        file_path = join((asd_path if is_asd else td_path), file_name)
        df        = pd.read_csv(file_path)
        df        = df.rename(mapper={"Idx": "fixation_id"}, axis=1)
        df        = df.rename(mapper={k: k.strip() for k in df.columns}, axis=1)
        transition = (df["fixation_id"].values == 0)
        df["participant_id"] = np.cumsum(transition)
        df["image_id"]       = image_id
        df["file_name"]      = file_name
        df["is_asd"]         = is_asd
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)
