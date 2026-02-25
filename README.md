# Churn Prediction App

A Streamlit-based application for predicting customer churn using machine learning.
The app includes RFM (Recency, Frequency, Monetary) analysis, hybrid clustering, and SHAP explainability.

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

## Usage

- Upload a CSV in the sidebar and map columns if names differ.
- Select the number of clusters in the sidebar.
- Explore tabs for RFM, distributions, clustering, churn prediction (cluster-based), and SHAP explanations.

## Notes

- For large files, consider sampling before upload.
- SHAP can be slow; reduce sample size in the sidebar.

## Technologies Used

- Python
- Streamlit
- Machine Learning Libraries (scikit-learn, pandas, etc.)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
