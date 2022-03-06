# Code for Arabic Dialect Identification 

## Setup
Install Dependencies
```
pip install -r requirements.txt
```

## Data Fetching

```
python data_fetching.py
```

This will fetch the labels from the API and create a new labeled csv file 

## Classical Model

Run  `classical_model.ipynb` to train a linear SVM model based on TF-IDF features


## DL Model

Run `MARBERT_Fine_Tuning.ipynb` to fine tune BERT on the dataset


