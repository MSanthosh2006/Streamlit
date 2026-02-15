import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

st.title("PCA + DBSCAN Clustering App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < 2:
        st.warning("Need at least 2 numeric columns.")
    else:
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.write("Explained Variance Ratio:")
        st.write(pca.explained_variance_ratio_)

        # DBSCAN Parameters
        eps = st.slider("Select eps", 0.1, 2.0, 0.5)
        min_samples = st.slider("Select min_samples", 2, 20, 5)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_pca)

        st.write("Number of Clusters:",
                 len(set(clusters)) - (1 if -1 in clusters else 0))
        st.write("Noise Points:", list(clusters).count(-1))

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA + DBSCAN Result")

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start.")
