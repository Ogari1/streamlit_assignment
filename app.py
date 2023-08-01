import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris Data
def iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(url, header=None, names=column_names)
    return data

data = iris_data()

st.title("Iris Dataset Explorer")
st.write("This is an interactive dataset explorer")

# Display the raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(data)

# Show the average sepal length for each species
if st.checkbox("Show the average sepal length for each species"):
    st.subheader("Average Sepal Length for each Species")
    avg_sepal_length = data.groupby("species")["sepal_length"].mean()
    st.write(avg_sepal_length)

# Display a scatter plot comparing two features
st.subheader("Compare two features using a scatter plot")
feature_1 = st.selectbox("Select the first feature:", data.columns[:-1])
feature_2 = st.selectbox("Select the second feature:", data.columns[:-1])

scatter_plot = px.scatter(data, x=feature_1, y=feature_2, color="species", hover_name="species")
st.plotly_chart(scatter_plot)

# Filter data based on species
st.subheader("Filter data based on species")
selected_species = st.multiselect("Select the species to display:", data["species"].unique())

if selected_species:
    filtered_data = data[data["species"].isin(selected_species)]
    st.dataframe(filtered_data)
else:
    st.write("No species selected.")

# Display a pairplot for the selected species
if st.checkbox("Show pairplot for the selected species"):
    st.subheader("Pairplot for the Selected Species")

    if selected_species:
        sns.pairplot(filtered_data, hue="species")
    else:
        sns.pairplot(data, hue="species")
        
    st.pyplot()

    # Show the distribution of a selected feature
st.subheader("Distribution of a Selected Feature")
selected_feature = st.selectbox("Select a feature to display its distribution:", data.columns[:-1])

hist_plot = px.histogram(data, x=selected_feature, color="species", nbins=30, marginal="box", hover_data=data.columns)
st.plotly_chart(hist_plot)


