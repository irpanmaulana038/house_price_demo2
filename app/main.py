from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib
import numpy as np

from io import BytesIO
import traceback

from app.features import create_features  # <-- pastikan file features.py benar

app = FastAPI()

# Load pipeline model (sudah termasuk preprocessing + model)
model = joblib.load("models/final_model.pkl")

@app.post("/predict_csv")
async def predict_from_csv(
    file: UploadFile = File(...),
    include_metrics: bool = Query(False, description="Set True untuk evaluasi jika ada kolom SalePrice")
):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Simpan target jika ingin evaluasi
        y_true = None
        if include_metrics and "SalePrice" in df.columns:
            y_true = df["SalePrice"]
            df = df.drop(columns=["SalePrice"])

        # Buat fitur tambahan
        df = create_features(df)

        # Prediksi
        y_pred = model.predict(df)
        result = {"predictions": y_pred.tolist()}

        # Evaluasi jika diminta
        if include_metrics and y_true is not None:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(np.log(y_pred),np.log1p(y_pred)))
            r2 = r2_score(y_true, y_pred)
            result["metrics"] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4)
            }

        return result

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"Internal error: {str(e)}"})
