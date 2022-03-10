import json

import pandas as pd
from sklearn.model_selection import train_test_split


def create_label2ind_file(labels):
    mapping = {}
    labels.sort()
    for idx in range(0, len(labels)):
        mapping[labels[idx]] = idx
    with open("labels.json", "w") as json_file:
        json.dump(mapping, json_file)
    return mapping


if __name__ == "__main__":
    RS = 17
    data = pd.read_csv("labeled_dataset.csv", lineterminator="\n")
    mapping = create_label2ind_file(data["dialect"].unique())
    data["dialect"].replace(mapping, inplace=True)
    print(data["dialect"].unique())
    train_df, val_df = train_test_split(
        data, test_size=0.2, random_state=RS, stratify=data["dialect"]
    )
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
