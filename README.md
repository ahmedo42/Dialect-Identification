# Code for Arabic Dialect Identification 

Install Dependencies
```
pip install -r requirements.txt
```

### Data Fetching

```
python data_fetching.py
```

This will fetch the labels from the API and create a new labeled csv file 

### Classical Model

to train a linear SVM model based on TF-IDF features

```
python classical_model.py
```


### DL Model

to fine tune MARBERT 

```
python fine_tune.py
```

