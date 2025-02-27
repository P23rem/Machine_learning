import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
def load_model():
    with open(r"Linear_regression\Student_app\student_performance.pkl", 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# Preprocess user input
def preprocessing_data(data, scaler):
    df = pd.DataFrame([data])  # Corrected DataFrame syntax
    df_transformed = scaler.transform(df)
    return df_transformed

# Predict performance
def predict_data(data):
    model, scaler = load_model()
    X_preprocessed = preprocessing_data(data, scaler)
    y_pred = model.predict(X_preprocessed)
    return y_pred[0]  # Return single value

# Streamlit app
def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to predict your performance")

    # Example input fields (customize based on your dataset)
    previous_scores = st.number_input("Previous Exam Score", min_value=0, max_value=100, step=1)

    if st.button("Predict"):
        data = {"Previous Scores": previous_scores}
        prediction = predict_data(data)
        st.success(f"Predicted Performance Score: {prediction:.2f}")

if __name__ == '__main__':
    main()
