import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load data with error handling
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")
        return None

# Train the model
@st.cache_data
def train_model(data):
    X = data.drop('quality', axis=1)
    y = data['quality'].apply(lambda x: 'GOOD' if x >= 7 else ('AVERAGE' if x >= 5 else 'BAD'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Wine Quality Prediction")

    # Load the data
    data = load_data("winequality-red.csv")

    # Check if data is loaded successfully
    if data is not None:
        # Display the dataset
        st.subheader("Wine Quality Dataset")
        st.write(data)

        # Train the model
        model = train_model(data)

        # User input for wine features
        st.sidebar.header("Input Features")
        fixed_acidity = st.sidebar.slider("Fixed Acidity", data['fixed acidity'].min(), data['fixed acidity'].max(), data['fixed acidity'].mean())
        volatile_acidity = st.sidebar.slider("Volatile Acidity", data['volatile acidity'].min(), data['volatile acidity'].max(), data['volatile acidity'].mean())
        citric_acid = st.sidebar.slider("Citric Acid", data['citric acid'].min(), data['citric acid'].max(), data['citric acid'].mean())
        residual_sugar = st.sidebar.slider("Residual Sugar", data['residual sugar'].min(), data['residual sugar'].max(), data['residual sugar'].mean())
        chlorides = st.sidebar.slider("Chlorides", data['chlorides'].min(), data['chlorides'].max(), data['chlorides'].mean())
        free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", data['free sulfur dioxide'].min(), data['free sulfur dioxide'].max(), data['free sulfur dioxide'].mean())
        total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", data['total sulfur dioxide'].min(), data['total sulfur dioxide'].max(), data['total sulfur dioxide'].mean())
        density = st.sidebar.slider("Density", data['density'].min(), data['density'].max(), data['density'].mean())
        pH = st.sidebar.slider("pH", data['pH'].min(), data['pH'].max(), data['pH'].mean())
        sulphates = st.sidebar.slider("Sulphates", data['sulphates'].min(), data['sulphates'].max(), data['sulphates'].mean())
        alcohol = st.sidebar.slider("Alcohol", data['alcohol'].min(), data['alcohol'].max(), data['alcohol'].mean())

        # Make prediction
        input_data = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        # Display prediction
        st.subheader("Prediction")
        st.write("Predicted Wine Quality:", prediction[0])

if __name__ == "__main__":
    main()
