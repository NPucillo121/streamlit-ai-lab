"""
Streamlit app for exploratory data analysis of the Iris dataset.
Allows users to visualize data distributions and relationships.
"""

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go


# Configure Streamlit page
st.set_page_config(
    page_title="Iris EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Iris Dataset - Exploratory Data Analysis")
st.markdown(
    "Explore the Iris dataset with interactive visualizations. "
    "Select columns to analyze distributions and relationships."
)

# Load the Iris dataset
iris_dataset = load_iris()
iris_df = pd.DataFrame(
    data=iris_dataset.data,
    columns=iris_dataset.feature_names
)
iris_df['species'] = iris_dataset.target_names[iris_dataset.target]

# Display dataset information
st.header("📋 Dataset Overview")

# Create two columns for dataset info
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", len(iris_df))

with col2:
    st.metric("Total Features", len(iris_dataset.feature_names))

with col3:
    st.metric("Species Classes", iris_df['species'].nunique())

# Display first rows of the dataset
st.subheader("First 10 Rows")
st.dataframe(iris_df.head(10), use_container_width=True)

# Display summary statistics
st.subheader("Summary Statistics")
st.dataframe(iris_df.describe(), use_container_width=True)

# Section for visualizations
st.header("📈 Data Visualization")

# Get list of numeric columns for user selection
numeric_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Create sidebar for user selections
st.sidebar.header("🎛️ Visualization Settings")

# Histogram settings
st.sidebar.subheader("Histogram Settings")
selected_histogram_column = st.sidebar.selectbox(
    "Select a column for histogram:",
    options=numeric_columns,
    key="histogram_column"
)

histogram_bins = st.sidebar.slider(
    "Number of bins:",
    min_value=5,
    max_value=50,
    value=20,
    step=1,
    key="histogram_bins"
)

# Scatter plot settings
st.sidebar.subheader("Scatter Plot Settings")
scatter_x_column = st.sidebar.selectbox(
    "Select X-axis column:",
    options=numeric_columns,
    key="scatter_x"
)

scatter_y_column = st.sidebar.selectbox(
    "Select Y-axis column:",
    options=numeric_columns,
    index=1 if len(numeric_columns) > 1 else 0,
    key="scatter_y"
)

# Create columns for visualizations
viz_col1, viz_col2 = st.columns(2)

# Display histogram
with viz_col1:
    st.subheader(f"📊 Histogram: {selected_histogram_column}")
    histogram = px.histogram(
        iris_df,
        x=selected_histogram_column,
        nbins=histogram_bins,
        color="species",
        barmode="overlay",
        title=f"Distribution of {selected_histogram_column}",
        labels={selected_histogram_column: selected_histogram_column, "count": "Frequency"},
        opacity=0.7
    )
    histogram.update_layout(
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(histogram, use_container_width=True)

# Display scatter plot
with viz_col2:
    st.subheader(f"🔍 Scatter Plot: {scatter_x_column} vs {scatter_y_column}")
    scatter = px.scatter(
        iris_df,
        x=scatter_x_column,
        y=scatter_y_column,
        color="species",
        title=f"{scatter_x_column} vs {scatter_y_column}",
        labels={
            scatter_x_column: scatter_x_column,
            scatter_y_column: scatter_y_column
        },
        hover_data={col: ":.2f" for col in numeric_columns}
    )
    scatter.update_layout(
        hovermode="closest",
        height=500
    )
    st.plotly_chart(scatter, use_container_width=True)

# Display correlation analysis
st.header("📊 Correlation Analysis")
st.subheader("Correlation Matrix Heatmap")

# Calculate correlation matrix
correlation_matrix = iris_df[numeric_columns].corr()

# Create heatmap
heatmap = go.Figure(
    data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale="RdBu",
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%.2f",
        textfont={"size": 10}
    )
)
heatmap.update_layout(
    title="Feature Correlation Matrix",
    height=600,
    width=700
)
st.plotly_chart(heatmap, use_container_width=True)

# Display species distribution
st.header("📌 Species Distribution")
species_counts = iris_df['species'].value_counts()
species_chart = px.pie(
    values=species_counts.values,
    names=species_counts.index,
    title="Distribution of Iris Species",
    hole=0.4
)
species_chart.update_layout(height=400)
st.plotly_chart(species_chart, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "📚 **Dataset Source:** UCI Machine Learning Repository | "
    "🔧 **Built with:** Streamlit, Plotly, and scikit-learn"
)
