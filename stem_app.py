import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from src.utils import load_pickle_file
from src.logger import logging

# Load model and preprocessor
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

model = load_pickle_file(model_path)
preprocessor = load_pickle_file(preprocessor_path)

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
            icons=['file-earmark-text', 'check-circle', 'info-circle'],  # Changed 'activity' to 'check-circle'
            menu_icon='iris-fill',  # Use 'iris-fill' as the menu icon
            default_index=0
        )


    # Navigation logic based on selected option
    if selected == "Home":
        st.title("Welcome to the Iris Species Classification App")
        
        # Create three columns for the images
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(r'G:\GIT_Project-2025\iris_project\static\images\Iris_setosa.jpg', caption="Iris Setosa")
        
        with col2:
            st.image(r'G:\GIT_Project-2025\iris_project\static\images\Iris-versicolor.jpg', caption="Iris Versicolor")
        
        with col3:
            st.image(r'G:\GIT_Project-2025\iris_project\static\images\Iris-virginica.jpg', caption="Iris Virginica")

        # Prompt the user
        st.write("Iris Setosa, or 'Sitka iris', is native to North America's wetlands.")
        st.write(" Iris Versicolor,Known as the 'Blue Flag iris', Iris Versicolor is found in wetland areas of North America.")
        st.write("Iris Virginica, or 'Virginia iris', is native to the eastern United States. ") 
        st.write("Its resilience and attractive foliage make it popular in gardens.")

        st.write("Are you intereted to knowing it classification?? Please got to 'Prediction' section.")

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
                species_images = {
                    'Iris-setosa': r'G:\GIT_Project-2025\iris_project\static\images\Iris_setosa.jpg',
                    'Iris-versicolor': r'G:\GIT_Project-2025\iris_project\static\images\Iris-versicolor.jpg',
                    'Iris-virginica': r'G:\GIT_Project-2025\iris_project\static\images\Iris-virginica.jpg'
                }

                st.success(f"The predicted species is: {predicted_species}")
                st.image(species_images[predicted_species], caption=predicted_species)

            except ValueError as ve:
                logging.error(f"Value error during prediction: {ve}")
                st.error("Value error: An issue occurred during prediction. Please check the input values.")
            except IndexError as ie:
                logging.error(f"Index error during prediction: {ie}")
                st.error("Index error: An unexpected issue occurred. Please try again.")
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                st.error("An error occurred during prediction. Please try again.")

    elif selected == "Author":
        st.title("Author Information")
        st.write("Md. Harun-Or-Rashid Khan")
        st.write("Data Science Expert")
        st.write("Experienced data scientist with ten years of teaching background. Proficient in Python, specializing in NLP, Generative AI, and using Streamlit for interfaces. Skilled in AWS and Linux. Deeply passionate about data analysis and machine learning. Seeking remote opportunities in Western time zones, adept at Git and GitHub collaboration.")

# Run the app
if __name__ == "__main__":
    main()





