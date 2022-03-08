from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from preprocessing import preprocess
from joblib import dump
import pandas as pd



if __name__ == "__main__":
    RS = 17
    MAX_FEATURES = 50000

    train_df = pd.read_csv("train.csv",lineterminator='\n')
    val_df = pd.read_csv("validation.csv",lineterminator='\n')

    train_df.loc[:, "text"] = train_df["text"].apply(preprocess)
    val_df.loc[:, "text"] = val_df["text"].apply(preprocess)

    tf_idf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(3, 7),analyzer="char")
    X_train, y_train = tf_idf.fit_transform(train_df.text), train_df.dialect
    X_val, y_val = tf_idf.transform(val_df.text), val_df.dialect

    clf = LinearSVC(random_state=RS, verbose=1,C=0.5)
    clf.fit(X_train, y_train)

    train_score = f1_score(y_train, clf.predict(X_train),average="macro")
    validation_score = f1_score(y_val, clf.predict(X_val),average="macro")

    print(f"Training Set F1 Score: {train_score}")
    print(f"Validation Set F1 Score: {validation_score}")
    dump(clf, "linear_svm.joblib") 