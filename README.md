# Diabetes Prediction App

A Streamlit-based web application for predicting diabetes risk using machine learning.

## Features

- Interactive web interface for diabetes prediction
- Uses XGBoost machine learning model
- Real-time prediction with progress animation
- User-friendly input sliders and selectors
- Responsive design

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd diabetes-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file:
   - `diabetes_model_XGB.pkl` should be in the root directory

## Running the App

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically

#### Option 2: Heroku
1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy to Heroku

#### Option 3: Docker
1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `diabetes_model_XGB.pkl` exists in the root directory
2. **Dependency Issues**: Use the provided `requirements.txt` with compatible versions
3. **Port Issues**: The app runs on port 8501 by default

### Model File
- The app expects a trained XGBoost model saved as `diabetes_model_XGB.pkl`
- The model should accept 11 features in this order:
  - age, hypertension, heart_disease, bmi, HbA1c_level
  - blood_glucose_level, gender_encoded
  - smoking_No_Info, smoking_current, smoking_former, smoking_never

## File Structure

```
diabetes-prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── diabetes_model_XGB.pkl  # Trained model file
├── README.md          # This file
└── Project .ipynb     # Jupyter notebook (training)
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).
