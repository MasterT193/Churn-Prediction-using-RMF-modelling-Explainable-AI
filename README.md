# Churn Prediction App

A Streamlit-based application for predicting customer churn using machine learning.

## Project Structure

- `churn_prediction_app/` - Main application directory
  - `app.py` - Streamlit application
  - `requirements.txt` - Python dependencies
- `online_retail_II.csv` - Dataset for training and analysis

## Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd Major_Web
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r churn_prediction_app/requirements.txt
```

## Running the Application

```bash
cd churn_prediction_app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Technologies Used

- Python
- Streamlit
- Machine Learning Libraries (scikit-learn, pandas, etc.)

## Continuous Integration

![CI](https://github.com/MasterT193/Churn-Prediction-using-RMF-modelling-Explainable-AI/actions/workflows/ci.yml/badge.svg)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
