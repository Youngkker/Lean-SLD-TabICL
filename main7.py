import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback
import numpy as np  # 导入 numpy 处理格式转换

try:
    from tabicl import TabICLClassifier

    print("✅ TabICL 库加载成功")
except ImportError:
    print("⚠️ 警告：未检测到 tabicl 库")

app = FastAPI(title="Lean SLD TabICL CDSS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# ⚠️ USER CONFIGURATION REQUIRED / 用户配置区 (MUST MODIFY BEFORE RUNNING)
# Please change the path below to the exact folder where you saved the .pkl files.
# 请将下方路径修改为您电脑上存放 .pkl 文件的实际文件夹绝对路径。
# Example for Windows: MODEL_DIR = r"C:\Users\Downloads\Lean-SLD-TabICL"
# Example for Mac/Linux: MODEL_DIR = "/Users/username/Downloads/Lean-SLD-TabICL"
# =====================================================================
MODEL_DIR = r"Please_Enter_Your_Model_Directory_Here"

model = None
imputer = None
features = None

CLINICAL_DEFAULTS = {
    'TG': 1.5, 'RBC': 4.5, 'UA': 350.0,
    'Glu': 5.2, 'BMI': 22.0, 'ALT': 25.0,
    'GGT': 30.0, 'LDL': 2.6
}


@app.on_event("startup")
def load_models():
    global model, imputer, features
    try:
        print(f"⏳ 正在加载临床模型组件...")
        imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
        model = joblib.load(os.path.join(MODEL_DIR, "Champion_TabICL.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
        print(f"✅ 引擎准备就绪！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")


class PatientData(BaseModel):
    Ethnicity: str
    BMI: float
    ALT: float
    TG: float = None
    UA: float = None
    Glu: float = None
    RBC: float = None
    GGT: float = None
    LDL: float = None


@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    try:
        # 使用 model_dump 兼容新版 Pydantic
        data = patient.model_dump() if hasattr(patient, 'model_dump') else patient.dict()

        raw_bmi = data['BMI']
        if data['Ethnicity'] == 'Asian':
            data['BMI'] = raw_bmi / 23.0
        elif data['Ethnicity'] == 'Western':
            data['BMI'] = raw_bmi / 25.0
        else:
            data['BMI'] = raw_bmi / 24.0

        final_input_row = {}
        for feat in features:
            val = data.get(feat)
            if val is None:
                final_input_row[feat] = CLINICAL_DEFAULTS.get(feat, 0.0)
            else:
                final_input_row[feat] = float(val)

        input_df = pd.DataFrame([final_input_row], columns=features)
        X_imp = imputer.transform(input_df)

        # 🌟 核心修复：使用 .item() 把 numpy 数字转为 Python 原生 float
        raw_prob = model.predict_proba(X_imp)[0][1]
        probability = float(raw_prob.item()) if hasattr(raw_prob, 'item') else float(raw_prob)

        return {
            "status": "success",
            "prediction": {
                "risk_probability": round(probability * 100, 2),
                "adjusted_bmi_ratio": float(round(data['BMI'], 4))
            }
        }
    except Exception as e:
        print(f"⚠️ 推理异常: \n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
