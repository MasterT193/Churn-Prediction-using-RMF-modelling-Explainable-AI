import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import hashlib
import re
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Customer Churn Prediction', page_icon='📊', layout='wide')
sns.set_theme(style='whitegrid', palette='deep')

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; }
    h1, h2, h3, h4 { color: #111827; }
    .stButton>button {
        background-color: #4F46E5;
        color: #FFFFFF;
        border: 0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTabs [role="tab"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        margin-right: 6px;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #EEF2FF;
        border-color: #C7D2FE;
    }
    .stMetric {
        background: #F8FAFC;
        padding: 0.75rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_db_connection():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'users.db')
    return sqlite3.connect(db_path, check_same_thread=False)


def init_user_table():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create a base table if missing, then migrate columns as needed
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    # Migration: ensure required columns exist for older DBs
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_user(username: str, full_name: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    # Detect current columns to avoid schema mismatch
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    # Ensure required columns exist
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        cur.execute('PRAGMA table_info(users)')
        cols = [row[1] for row in cur.fetchall()]
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
        cur.execute('PRAGMA table_info(users)')
        cols = [row[1] for row in cur.fetchall()]
    conn.commit()
    try:
        cur.execute(
            'INSERT INTO users (username, full_name, password_hash) VALUES (?, ?, ?)',
            (username.strip().lower(), full_name.strip(), hash_password(password)),
        )
        conn.commit()
        return True, 'Registration successful. You can now log in.'
    except sqlite3.IntegrityError:
        return False, 'Username already exists. Please choose a different one.'
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure columns exist for older DBs
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    conn.commit()
    cur.execute(
        'SELECT id, username, full_name FROM users WHERE username = ? AND password_hash = ?',
        (username.strip().lower(), hash_password(password)),
    )
    row = cur.fetchone()
    conn.close()
    return row


def valid_username(username: str) -> bool:
    return bool(re.fullmatch(r'[A-Za-z0-9_.-]{4,30}', username or ''))


def valid_password(password: str) -> bool:
    if password is None:
        return False
    has_min_len = len(password) >= 8
    has_alpha = bool(re.search(r'[A-Za-z]', password))
    has_digit = bool(re.search(r'\d', password))
    return has_min_len and has_alpha and has_digit


init_user_table()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'full_name' not in st.session_state:
    st.session_state.full_name = ''

st.title('Customer Churn Prediction System')

if not st.session_state.logged_in:
    st.subheader('Login Required')
    auth_tab1, auth_tab2 = st.tabs(['Login', 'Register'])

    with auth_tab1:
        with st.form('login_form'):
            login_username = st.text_input('Username')
            login_password = st.text_input('Password', type='password')
            login_btn = st.form_submit_button('Login')

        if login_btn:
            user = authenticate_user(login_username, login_password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.full_name = user[2]
                st.success(f'Welcome back, {user[2]}!')
                st.rerun()
            else:
                st.error('Invalid username or password.')

    with auth_tab2:
        with st.form('register_form'):
            reg_full_name = st.text_input('Full Name')
            reg_username = st.text_input('Username (4-30 chars, letters/numbers/._-)')
            reg_password = st.text_input('Password (min 8 chars, include letters and numbers)', type='password')
            reg_confirm_password = st.text_input('Confirm Password', type='password')
            reg_btn = st.form_submit_button('Create Account')

        if reg_btn:
            if not reg_full_name.strip():
                st.error('Full name is required.')
            elif not valid_username(reg_username):
                st.error('Invalid username format.')
            elif not valid_password(reg_password):
                st.error('Password must be at least 8 characters and include letters and numbers.')
            elif reg_password != reg_confirm_password:
                st.error('Passwords do not match.')
            else:
                ok, msg = create_user(reg_username, reg_full_name, reg_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.stop()


# Sidebar with project info and file upload
st.sidebar.title('Churn Prediction System')
st.sidebar.info('Upload your customer data and explore churn risk using unsupervised learning and explainable AI.')
st.sidebar.success(f"Logged in as: {st.session_state.full_name}")
if st.sidebar.button('Logout'):
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.full_name = ''
    st.rerun()
st.sidebar.markdown('---')
st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader('Upload your customer data (CSV)', type=['csv'])
st.sidebar.markdown('---')
st.sidebar.write('Developed with ❤️ using Streamlit')

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write('### Raw Data', data.head())
    st.write('#### Columns in your data:', list(data.columns))

    required_cols = ['Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.warning("Please upload a CSV with the required columns or update the code to match your data.")
    else:
        # User selects number of clusters
        n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=8, value=4, help='Choose how many customer segments to create')

        # Tabs for each analysis section
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'RFM Analysis', 'RFM Distributions', 'Clustering', 'Churn Prediction', 'Explainable AI (SHAP)'])

        with tab1:
            st.write('## RFM Analysis')
            st.info('RFM (Recency, Frequency, Monetary) analysis segments customers based on how recently, how often, and how much they purchase. This helps identify valuable and at-risk customers.')
            data['TotalPrice'] = data['Quantity'] * data['Price']
            # Ensure InvoiceDate is datetime
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
            rfm = data.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            st.dataframe(rfm.head())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Total Customers', len(rfm))
            c2.metric('Avg Recency', int(rfm['Recency'].mean()))
            c3.metric('Avg Frequency', round(rfm['Frequency'].mean(), 2))
            c4.metric('Avg Monetary', round(rfm['Monetary'].mean(), 2))

        with tab2:
            st.write('### RFM Feature Distributions')
            st.caption('These histograms show the distribution of Recency, Frequency, and Monetary values across all customers.')
            fig_rfm, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].hist(rfm['Recency'], bins=20, color='skyblue')
            axs[0].set_title('Recency')
            axs[1].hist(rfm['Frequency'], bins=20, color='lightgreen')
            axs[1].set_title('Frequency')
            axs[2].hist(rfm['Monetary'], bins=20, color='salmon')
            axs[2].set_title('Monetary')
            st.pyplot(fig_rfm)
            st.markdown(
                "- **Recency**: Lower values mean recent purchases; higher values indicate longer time since last purchase.\n"
                "- **Frequency**: Higher values mean more invoices (more frequent purchases).\n"
                "- **Monetary**: Higher values indicate greater total spending. These shapes show how customers are distributed across each RFM dimension."
            )

        with tab3:
            st.write('## Hybrid Clustering')
            st.info('Clustering groups customers with similar RFM profiles. KMeans and Agglomerative clustering are used to find natural segments in your customer base.')
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(rfm_scaled)
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            agg_labels = agg.fit_predict(rfm_scaled)
            rfm['KMeans_Cluster'] = kmeans_labels
            rfm['Agg_Cluster'] = agg_labels
            st.dataframe(rfm.head())
            st.write('### Cluster Scatterplot (Recency vs Monetary)')
            st.caption('Each point is a customer, colored by their cluster. This helps visualize how clusters separate based on Recency and Monetary value.')
            fig_scatter, ax_scatter = plt.subplots()
            scatter = ax_scatter.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['KMeans_Cluster'], cmap='tab10', alpha=0.7)
            legend1 = ax_scatter.legend(*scatter.legend_elements(), title="KMeans Cluster")
            ax_scatter.add_artist(legend1)
            ax_scatter.set_xlabel('Recency')
            ax_scatter.set_ylabel('Monetary')
            st.pyplot(fig_scatter)
            st.markdown(
                "This scatter shows customers by **Recency** (x-axis) and **Monetary** (y-axis), colored by KMeans cluster.\n"
                "- Points to the right (higher Recency) are less recent customers.\n"
                "- Points lower (smaller Monetary) spend less.\n"
                "Distinct color groups indicate segments with similar RFM behavior."
            )

        with tab4:
            st.write('## Churn Prediction (Cluster-based)')
            st.info('Customers in certain clusters (e.g., high Recency, low Frequency/Monetary) may be at higher risk of churn. This unsupervised approach uses clusters as a proxy for churn risk.')
            cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
            st.write('### Number of Customers per KMeans Cluster')
            st.caption('This bar chart shows how many customers are in each cluster.')
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values, color='orchid')
            ax_bar.set_xlabel('KMeans Cluster')
            ax_bar.set_ylabel('Number of Customers')
            st.pyplot(fig_bar)
            st.markdown(
                "This bar chart shows how many customers fall into each KMeans cluster.\n"
                "Use this to gauge the size of segments (e.g., whether the high-risk cluster is small and targeted or large and widespread)."
            )
            st.write(rfm.groupby('KMeans_Cluster').mean())

            cluster_profile = rfm.groupby('KMeans_Cluster')[['Recency', 'Frequency', 'Monetary']].mean().copy()
            rec_norm = (cluster_profile['Recency'] - cluster_profile['Recency'].min()) / (
                cluster_profile['Recency'].max() - cluster_profile['Recency'].min() + 1e-9
            )
            freq_norm = (cluster_profile['Frequency'] - cluster_profile['Frequency'].min()) / (
                cluster_profile['Frequency'].max() - cluster_profile['Frequency'].min() + 1e-9
            )
            mon_norm = (cluster_profile['Monetary'] - cluster_profile['Monetary'].min()) / (
                cluster_profile['Monetary'].max() - cluster_profile['Monetary'].min() + 1e-9
            )
            cluster_profile['RiskScore'] = rec_norm + (1 - freq_norm) + (1 - mon_norm)
            high_risk_cluster = int(cluster_profile['RiskScore'].idxmax())
            rfm['Predicted_Churn'] = np.where(rfm['KMeans_Cluster'] == high_risk_cluster, 'High Risk', 'Lower Risk')

            high_risk_customers = rfm[rfm['Predicted_Churn'] == 'High Risk'].copy()
            high_risk_count = len(high_risk_customers)
            total_customers = len(rfm)
            high_risk_pct = (high_risk_count / total_customers * 100) if total_customers else 0

            st.write('### Final Prediction Statement')
            st.success(
                f"Predicted churn type: **customer inactivity churn risk**. "
                f"Based on your uploaded data, cluster **{high_risk_cluster}** is the highest-risk segment. "
                f"Predicted at-risk customers: **{high_risk_count} / {total_customers} ({high_risk_pct:.2f}%)**."
            )
            st.caption(
                'Interpretation: High-risk customers are those with relatively higher recency (longer time since last purchase), '
                'and comparatively lower frequency and/or lower monetary value than other clusters.'
            )

            st.write('### High-Risk Customer List (Top 100)')
            st.dataframe(high_risk_customers[['Recency', 'Frequency', 'Monetary', 'KMeans_Cluster', 'Predicted_Churn']].head(100))

            # Download button for cluster results
            csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button('Download Clustered Data as CSV', csv, 'clustered_customers.csv', 'text/csv')

        with tab5:
            st.write('## Explainable AI (SHAP)')
            st.info('SHAP (SHapley Additive exPlanations) explains which RFM features are most important for assigning customers to clusters, helping you understand the drivers of churn risk.')
            explainer = shap.KernelExplainer(kmeans.predict, rfm_scaled)
            shap_values = explainer.shap_values(rfm_scaled[:50])
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, rfm.iloc[:50, :3], show=False)
            st.pyplot(fig)
            st.markdown(
                "The SHAP summary ranks features by their contribution to cluster assignment.\n"
                "- Larger absolute SHAP values mean stronger influence.\n"
                "- Color often reflects feature value (depends on SHAP plot style).\n"
                "Use this to understand whether **Recency**, **Frequency**, or **Monetary** drives segmentation and churn risk."
            )

        st.write('---')
        st.write('This is a demo. For production, tune clustering and RFM logic to your data.')
else:
    st.info('Awaiting CSV file upload.')
