import os.path

import pandas as pd
import numpy as np
import json

from os.path import join
from tqdm import tqdm

from eyefeatures.visualization.static_visualization import scanpath_visualization


def read_dataset(fixation_path: str, meta_path: str) -> pd.DataFrame:
    fixations_data = pd.read_excel(fixation_path)
    fixations_data["SubjectID"] = fixations_data["SubjectID"].astype(str)
    fixations_data["SentenceID"] = fixations_data["Sentence_ID"].astype(str)
    fixations_data.drop(["Sentence_ID"], axis=1, inplace=True)

    meta_data = pd.read_excel(meta_path)
    meta_data["Grade"] = meta_data["Grade"].astype(int).map(lambda x: 1 if x >= 3 else 0)
    meta_data["SubjectID"] = meta_data["SubjID"].astype(str)
    meta_data = meta_data.set_index(["SubjectID"])
    meta_data.drop(["SubjID"], axis=1, inplace=True)

    df = fixations_data.merge(meta_data, left_on="SubjectID", right_index=True, how="inner").reset_index(drop=True)
    df["GroupID"] = df[["SubjectID", "SentenceID"]].apply(lambda row: row["SubjectID"] + "_" + row["SentenceID"], axis=1)
    return df


def get_and_save_image_array(
    df: pd.DataFrame,
    group_id: str,
    save_path: str = None,
    x: str = "FIX_X",
    y: str = "FIX_Y",
    **kwargs
) -> np.ndarray:
    sub_df = df[df["GroupID"] == group_id]
    path_to_img = join(save_path, group_id) if save_path else None
    image  = scanpath_visualization(
        sub_df, x=x, y=y, return_ndarray=True,
        show_plot=False, path_to_img=path_to_img,
        **kwargs
    )
    return (image * 255).astype(np.uint8)


def dataset2images(fixation_path: str, meta_path: str, save_path: str, **kwargs):
    df = read_dataset(fixation_path, meta_path)
    group_ids = df.GroupID.unique()

    # get visualization for each image and save them
    for group_id in tqdm(group_ids):
        _ = get_and_save_image_array(df, group_id, join(save_path, "dyslexia_images"), **kwargs)
        return

    # save mapping GroupID -> Grade
    sub_df = df[["GroupID", "Grade"]]
    sub_df = sub_df.drop_duplicates(subset=["GroupID"])
    sub_df = sub_df.set_index(["GroupID"], drop=True)
    group_id2grade = sub_df.to_dict(orient="dict")
    group_id2grade = group_id2grade["Grade"]

    with open(join(save_path, "group_id2grade.json"), "w") as iofile:
        json.dump(group_id2grade, iofile)


if __name__ == "__main__":
    DATA_PATH = join("..", "data")

    FIXATION_PATH = join(DATA_PATH, "Fixation_report_dyslexia.xlsx")
    META_PATH     = join(DATA_PATH, "soc-dem.xlsx")
    SAVE_PATH     = DATA_PATH

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    kwargs = {
        "path_width": 1,
        "add_regressions": True,
        "regression_color": "red",
        "points_width": 50000,
        "rule": (2, 3)
    }
    dataset2images(FIXATION_PATH, META_PATH, SAVE_PATH, **kwargs)
