# California House Price Prediction

This project predicts California house prices using machine learning. It includes data analysis, model training, and a modern Streamlit web app for interactive predictions.

## Features
- Data analysis and visualization in Jupyter Notebook
- Model training with scikit-learn (RandomForestRegressor)
- Preprocessing pipeline for numerical and categorical features
- Model and pipeline persistence with joblib
- Streamlit web app for user-friendly predictions with custom styling

## Project Structure
```
MLHousePricePrediction/
├── housing.csv                # Dataset
├── AnalyzingData.ipynb        # Data analysis notebook
├── joblib_inference.py        # Model training & inference script
├── main.py                    # Example script (optional)
├── streamlit_app.py           # Streamlit web application
└── README.md                  # Project documentation
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   # or install manually:
   pip install pandas numpy scikit-learn matplotlib joblib streamlit
   ```
3. **Train the model**
   ```sh
   python joblib_inference.py
   ```
   This will create `model.pkl` and `pipeline.pkl`.
4. **Run the Streamlit app**
   ```sh
   streamlit run streamlit_app.py
   ```

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

## License
MIT License

---
*Created with ❤️ for California housing data science projects.*
