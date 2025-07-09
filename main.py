import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

DS = FastAPI()

model = joblib.load('ds_model.pkl')
scalar = joblib.load('ds_scaler.pkl')


class TumorFeatures(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float



@DS.post('/predict/')
async def predict(person: TumorFeatures):
    ds_dict = person.dict()

    features = list(ds_dict.values())
    scaled = scalar.transform([features])

    pred = model.predict(scaled)[0]
    print(model.predict(scaled))
    prob = model.predict_proba(scaled)[0]
    print(model.predict_proba(scaled))

    return {'diagnosis': pred, "probability": float(max(prob))}


if __name__ == "__main__":
    uvicorn.run(DS, host="127.0.0.1", port=8100)