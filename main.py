from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
from sklearn.linear_model import Lasso

with open('weights.pkl', 'rb') as f:
    weights = load(f)

model = Lasso()

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return 12345


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)
    r2_score(y_test, prediction)

    return model.predict(items)
