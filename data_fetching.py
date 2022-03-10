import json

import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("unlabeled_dataset.csv")
    data["id"] = data["id"].astype(str)
    data["text"] = "empty"
    request_size = 1000
    dataset_size = len(data)
    URL = "https://recruitment.aimtechnologies.co/ai-tasks"
    ids = data["id"].values.astype(str).tolist()
    text = []
    for i in range(0, dataset_size, request_size):
        final = min(i + request_size, len(data))
        payload = ids[i:final]
        response = requests.post(URL, data=json.dumps(payload)).json()
        text.extend(response.values())
    data["text"] = text
    data.to_csv("labeled_dataset.csv", index=False)
