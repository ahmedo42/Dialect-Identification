# Code for Arabic Dialect Identification 

Training and deployment of Arabic DI models using [MARBERT](https://huggingface.co/UBC-NLP/MARBERT) and [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
### Scripts Overview

    ├── requirements.txt            # Dependencies for training and deployment
    |── data_fetching.py            # Script for fetching the dataset labels from a REST API
    ├── create_datasets.py          # Splits labeled_dataset.csv into train.csv and validation.csv
    |── preprocessing.py            # Text preprocessing for the models
    ├── classical_model.py          # Linear SVM model trained using TFIDF features
    ├── fine_tune.py                # Fine Tunes MARBERT model
    ├── deploy.py                   # Deploys the two models locally on seperate routes for inference

### Usage

- Install the requiremnts
```
pip install -r requirements.txt
```

- Fetch the labels 

```
python data_fetching.py

# or download the labeled dataset directly from google drive

gdown --id 1ir0WJRJQPBVihJ4fkc-ztewV84nyxtgt
```

- Split the dataset into training and validation sets (80/20 split)

```
python create_datasets.py
```

- Train the SVM model

```
python classical_model.py
```

- Fine-Tune the MARBERT model

```
python fine_tune.py
```

- To download the weights both models

```
pip install --upgrade --no-cache-dir gdown
gdown --id 1-NGYSdCjFU4-tWHwakKWYXUYOKK9ux8r # MARBERT checkpoint
gdown --id 1-6JVk3-fR3Q0MGkYQstfMP8fqi090YBh # TFIDF trained features
gdown --id 1yBHxGfRpn8XZswQuU4_7hAyOyjarn47t # Trained SVM model
```

- To deploy the models for inference , simply run

```
python deploy.py
```
- To send requests for the server using `httpie` module for example , use `predict` for DL model and  `predict_classical` for SVM model

```
http POST the_public_ngrok_url/predict text=":D شوف بقي  اللي انت عايزه"
```

### Refrences 
- [QADI Paper](https://arxiv.org/pdf/2005.06557.pdf)
- [MARBERT Paper](https://arxiv.org/abs/2101.01785)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pytorch-Lightning](https://www.pytorchlightning.ai/)