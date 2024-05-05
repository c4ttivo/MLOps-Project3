from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import mlflow.sklearn
from sklearn.compose import make_column_transformer
import pandas as pd


os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

mlflow.set_tracking_uri("http://mlflow_server:5000/")
model = mlflow.sklearn.load_model("models:/random_forest@production")


class Covertype(BaseModel):    
    Elevation: int = 0
    Aspect: int = 0
    Slope: int = 0
    Horizontal_Distance_To_Hydrology: int = 0
    Vertical_Distance_To_Hydrology: int = 0
    Horizontal_Distance_To_Roadways: int = 0
    Hillshade_9am: int = 0
    Hillshade_Noon: int = 0
    Hillshade_3pm: int = 0    
    Horizontal_Distance_To_Fire_Points: int = 0
    Wilderness_Area: str = "Cache"
    Soil_Type: str = "C2702"
    Cover_Type: int = 0

    
app = FastAPI()

@app.get("/")
async def root():
    
    return {"message": "Convet Type Inference API"}

@app.post("/predict/")
def predict_species_random_forest(covertype: Covertype):

    data = {
        "Elevation": [covertype.Elevation],
        "Aspect": [covertype.Aspect],
        "Slope": [covertype.Slope],
        "Horizontal_Distance_To_Hydrology": [covertype.Horizontal_Distance_To_Hydrology],
        "Vertical_Distance_To_Hydrology": [covertype.Vertical_Distance_To_Hydrology],
        "Horizontal_Distance_To_Roadways": [covertype.Horizontal_Distance_To_Roadways],
        "Hillshade_9am": [covertype.Hillshade_9am],
        "Hillshade_Noon": [covertype.Hillshade_Noon],
        "Hillshade_3pm": [covertype.Hillshade_3pm],
        "Horizontal_Distance_To_Fire_Points": [covertype.Horizontal_Distance_To_Fire_Points],
        "Wilderness_Area": [covertype.Wilderness_Area],
        "Soil_Type": [covertype.Soil_Type]
    }

    df_to_predict = pd.DataFrame(data)

    # Hacer predicciones
    prediction = model.predict(df_to_predict)
    res = prediction[0].item()

    return{"Predicted cover_type": res}
    
