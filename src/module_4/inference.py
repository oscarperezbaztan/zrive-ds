import logging
import os
from joblib import load
from train import *
from utils import *

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

def main():
    model_name = "20240217-201103_XGBoost_0.25.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info(f"Loaded model {model_name}")

    df = preprocess_data_manually()
    X, y = feature_label_split(FEATURE_COLS, TARGET,df)

    y_pred = model.predict_proba(X)[:,1]

    evaluate_model("Inference test", y, y_pred)

if __name__ == "__main__":
    main()