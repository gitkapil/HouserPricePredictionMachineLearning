
# California House Price Prediction (Python Project)

This is a Python machine learning project for predicting California house prices. It features model training, inference, and a modern Streamlit web app for interactive predictions.

## Features
- Model training with scikit-learn (RandomForestRegressor)
- Preprocessing pipeline for numerical and categorical features
- Model and pipeline persistence with joblib
- Streamlit web app for user-friendly predictions
- (Optional) Data analysis and visualization in Jupyter Notebook (see `notebooks/`)

## Project Structure
```
HouserPricePredictionMachineLearning/
├── streamlit_app.py        # Streamlit web application (main entry point)
├── joblib_inference.py     # Model training & inference script
├── train_models.py         # Additional training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── model.pkl               # Trained model (Git LFS)
├── pipeline.pkl            # Preprocessing pipeline (Git LFS)
├── housing.csv             # (Optional) Dataset for training
└── notebooks/
    └── AnalyzingData.ipynb # Data analysis notebook (optional)
```

## Quick Start
1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Train the model (if needed)**
   ```sh
   python joblib_inference.py
   ```
   This will create `model.pkl` and `pipeline.pkl`.
4. **Run the Streamlit app**
   ```sh
   streamlit run streamlit_app.py
   ```

## Notes
- The main entry point is `streamlit_app.py` (Python script).
- Notebooks are for optional data exploration and are not required to use the app.
- All core logic and deployment are in Python scripts.

## Usage
- Open the Streamlit app in your browser.
- Enter house features in the form.
- Click "Predict House Value" to see the predicted price on the right.

## Customization
- Update `streamlit_app.py` for UI changes or new features.
- Retrain the model with new data by rerunning `joblib_inference.py`.

## Requirements
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, joblib, streamlit

