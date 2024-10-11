import os
import sys
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from src.utils import load_pickle_file
from src.logger import logging
from src.exception import CustomError

# Get the current working directory
current_dir = os.path.dirname(__file__)

# Load model and preprocessor using absolute paths
model_path = os.path.join(current_dir, 'artifacts', 'model.pkl')
preprocessor_path = os.path.join(current_dir, 'artifacts', 'preprocessor.pkl')

# Check if files exist
if not os.path.exists(model_path):
    logging.info(f"Model file not found: {model_path}")
    st.info("Model file not found. Please ensure it exists.")

if not os.path.exists(preprocessor_path):
    logging.info(f"Preprocessor file not found: {preprocessor_path}")
    st.info("Preprocessor file not found. Please ensure it exists.")

# Load model and preprocessor
model = load_pickle_file(model_path)
preprocessor = load_pickle_file(preprocessor_path)

# Define paths for the images
image_dir = os.path.join(current_dir, 'static', 'images')

# Define species images paths
species_images = {
    'Iris-setosa': os.path.join(image_dir, 'Iris_setosa.jpg'),
    'Iris-versicolor': os.path.join(image_dir, 'Iris-versicolor.jpg'),
    'Iris-virginica': os.path.join(image_dir, 'Iris-virginica.jpg')
}

# Set page configuration
st.set_page_config(page_title="Iris Classification App",
                   layout="wide",
                   page_icon="🌷")

# Define the Streamlit app
def main():
    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            'Classification App',
            ['Home', 'Predict', 'Author'],
            icons=['file-earmark-text', 'check-circle', 'info-circle'],
            menu_icon='iris-fill',
            default_index=0
        )

    # Navigation logic based on selected option
    if selected == "Home":
        st.title("Welcome to the Iris Species Classification App")

        # Create three columns for the images
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(species_images['Iris-setosa'], caption="Iris Setosa")
        
        with col2:
            st.image(species_images['Iris-versicolor'], caption="Iris Versicolor")
        
        with col3:
            st.image(species_images['Iris-virginica'], caption="Iris Virginica")

        # Prompt the user
        st.write("Iris Setosa, or 'Sitka iris', is native to North America's wetlands.")
        st.write("Iris Versicolor, known as the 'Blue Flag iris', is found in wetland areas of North America.")
        st.write("Iris Virginica, or 'Virginia iris', is native to the eastern United States.")
        st.write("Its resilience and attractive foliage make it popular in gardens.")

        st.write("Are you interested in knowing its classification? Please go to the 'Prediction' section.")

    elif selected == "Predict":
        st.title("Iris Species Prediction")
        
        # Form input
        st.header("Enter the features of the Iris flower")
        SepalLengthCm = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
        SepalWidthCm = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)
        PetalLengthCm = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
        PetalWidthCm = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)

        if st.button("Predict"):
            try:
                # Create DataFrame from input
                input_data = pd.DataFrame([{
                    'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm
                }])

                # Preprocess the input data
                X_new = preprocessor.transform(input_data)

                # Make prediction
                prediction = model.predict(X_new)[0]
                species_index = int(round(prediction))  # Ensure the prediction is an integer index

                # Store prediction in session state
                st.session_state.predicted_species = species_index
                st.session_state.input_data = input_data

                # Display result
                species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                predicted_species = species_names[species_index]

                # Show species image
                st.success(f"The predicted species is: {predicted_species}")
                st.image(species_images[predicted_species], caption=predicted_species)

            except Exception as e:
                raise CustomError(str(e), sys)

    elif selected == "Author":
        st.title("Author Information")
        st.write("Md. Harun-Or-Rashid Khan")
        st.write("Data Science Expert")
        st.write("Experienced data scientist with ten years of teaching background. Proficient in Python, specializing in NLP, Generative AI, and using Streamlit for interfaces. Skilled in AWS and Linux. Deeply passionate about data analysis and machine learning. Seeking remote opportunities in Western time zones, adept at Git and GitHub collaboration.")

# Run the app
if __name__ == "__main__":
    main()






