
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import logging


MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs),
        ('cat', cat_pipline, cat_attribs)
    ])

    return full_pipeline


try:
    if not os.path.exists(MODEL_FILE):
        logging.info("Training mode: Model file not found. Starting training...")
        try:
            housing = pd.read_csv("housing.csv")
        except Exception as e:
            logging.error(f"Failed to read housing.csv: {e}")
            raise

        # Create a stratified test set
        housing['income_cat'] = pd.cut(housing["median_income"],
                                    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(housing, housing['income_cat']):
            try:
                housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
            except Exception as e:
                logging.error(f"Failed to write input.csv: {e}")
                raise
            housing = housing.loc[train_index].drop("income_cat", axis=1)

        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_attribs = ["ocean_proximity"]

        pipeline = build_pipeline(num_attribs, cat_attribs)
        housing_prepared = pipeline.fit_transform(housing_features)

        model = RandomForestRegressor(random_state=42)
        model.fit(housing_prepared, housing_labels)

        try:
            joblib.dump(model, MODEL_FILE)
            joblib.dump(pipeline, PIPELINE_FILE)
        except Exception as e:
            logging.error(f"Failed to save model or pipeline: {e}")
            raise
        logging.info("Model is trained and saved. Congrats!")
    else:
        logging.info("Inference mode: Model file found. Starting inference...")
        try:
            model = joblib.load(MODEL_FILE)
            pipeline = joblib.load(PIPELINE_FILE)
        except Exception as e:
            logging.error(f"Failed to load model or pipeline: {e}")
            raise

        try:
            input_data = pd.read_csv('input.csv')
        except Exception as e:
            logging.error(f"Failed to read input.csv: {e}")
            raise
        try:
            transformed_input = pipeline.transform(input_data)
            predictions = model.predict(transformed_input)
            input_data['median_house_value'] = predictions
            input_data.to_csv("output.csv", index=False)
        except Exception as e:
            logging.error(f"Failed during inference or saving output.csv: {e}")
            raise
        logging.info("Inference is complete, results saved to output.csv. Enjoy!")
except Exception as main_e:
    logging.error(f"An error occurred in the main workflow: {main_e}")