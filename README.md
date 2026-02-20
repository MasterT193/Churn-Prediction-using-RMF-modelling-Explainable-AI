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

Note: when cloning or deploying from GitHub, use the repository root URL (e.g. `https://github.com/<user>/<repo>`). A GitHub folder URL like `.../tree/main/churn_prediction_app` is not a valid repository URL.

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
