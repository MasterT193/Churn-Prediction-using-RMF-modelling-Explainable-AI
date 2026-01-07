import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import shap
import matplotlib.pyplot as plt
import seaborn as sns


# Sidebar with project info and file upload
st.sidebar.title('Churn Prediction System')
st.sidebar.info('Upload your customer data and explore churn risk using unsupervised learning and explainable AI.')
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
            st.metric('Total Customers', len(rfm))
            st.metric('Avg Recency', int(rfm['Recency'].mean()))
            st.metric('Avg Frequency', round(rfm['Frequency'].mean(), 2))
            st.metric('Avg Monetary', round(rfm['Monetary'].mean(), 2))

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
            st.write(rfm.groupby('KMeans_Cluster').mean())
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

        st.write('---')
        st.write('This is a demo. For production, tune clustering and RFM logic to your data.')
else:
    st.info('Awaiting CSV file upload.')
