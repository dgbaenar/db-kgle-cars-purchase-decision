import numpy as np
import joblib
import json
import pandas as pd

from sklearn import impute, linear_model, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import settings as st


def preprocess():
    df = pd.read_csv('./data/csv/raw.csv')
    df.columns = [column.upper().replace(' ', '_') for column in df.columns]

    to_test = df.sample(1)
    to_test = to_test.T.to_dict()[to_test.index[0]]

    with open("./data/tests/test,json", 'w') as outfile:
        json.dump(to_test, outfile, indent=4)

    if st.NUMERICAL_FEATURES:
        imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        df[st.NUMERICAL_FEATURES] = imputer.fit_transform(
            df[st.NUMERICAL_FEATURES])
        joblib.dump(imputer, "./data/joblib/imputer.joblib.dat")

    if st.CATEGORICAL_FEATURES:
        df[st.CATEGORICAL_FEATURES] = df[st.CATEGORICAL_FEATURES].apply(
            lambda x: x.astype(str).str.upper())

    # One hot encoding
    categorical = pd.get_dummies(df[st.CATEGORICAL_FEATURES])
    df = df.join(categorical)
    df = df.drop(st.CATEGORICAL_FEATURES, axis=1)

    return df


def split(data):
    X = data[st.FINAL_FEATURES]
    y = data.pop(st.DEPENDENT_VARIABLE)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y,
                                                                        test_size=st.TEST_SIZE,
                                                                        random_state=st.RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def training(X_train, X_test, y_train, y_test):
    model = linear_model.LogisticRegression(random_state=st.RANDOM_STATE)
    model.fit(X_train, y_train)
    joblib.dump(model, "./data/joblib/model.joblib.dat")

    y_pred = model.predict(X_test)

    print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
    print("Classification Report:\n {}".format(
        classification_report(y_test, y_pred)))


if __name__ == "__main__":
    data = preprocess()
    X_train, X_test, y_train, y_test = split(data)
    training(X_train, X_test, y_train, y_test)
