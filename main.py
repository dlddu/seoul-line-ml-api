from typing import List

import pandas
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATASET_FILE = "/data/dataset.csv"

app = FastAPI()
encoder = None
xgb = None


@app.post("/reload-dataset")
async def reload_dataset():
    global xgb, encoder
    train = pandas.read_csv(DATASET_FILE)

    X_train = train.drop("Get Off", axis=1)
    y_train = train["Get Off"]

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(X_train)
    X_train = encoder.transform(X_train).toarray()

    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)


@app.post("/predict")
async def predict(request: List[dict]):
    global xgb, encoder
    if xgb is None:
        return {"error": "Model not trained yet"}

    X = pandas.DataFrame.from_records(request)
    X = encoder.transform(X).toarray()

    predictions = xgb.predict(X).tolist()
    return [*predictions]


@app.get("/health")
async def health():
    return {"status": "ok2"}
