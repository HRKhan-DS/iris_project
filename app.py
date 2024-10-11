from flask import Flask, render_template, request, url_for, redirect
import os
from src.utils import load_pickle_file
from src.logger import logging
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load model and preprocessor
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

model = load_pickle_file(model_path)
preprocessor = load_pickle_file(preprocessor_path)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/form')
def form():
    return render_template('new_form.html')

@app.route('/predictform', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input for the features
            SepalLengthCm = float(request.form['SepalLengthCm'])
            SepalWidthCm = float(request.form['SepalWidthCm'])
            PetalLengthCm = float(request.form['PetalLengthCm'])
            PetalWidthCm = float(request.form['PetalWidthCm'])

            # Create a DataFrame from form data
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

            # Redirect to the result page with the prediction
            return redirect(url_for('result', species=prediction))
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return "An error occurred during prediction. Please try again."

    return render_template('new_form.html')

# Display prediction
@app.route('/result/<species>')
def result(species):
    species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    species_images = {
        'Iris-setosa': 'images/iris_setosa.jpg',  # Removed static/
        'Iris-versicolor': 'images/Iris-versicolor.jpg',
        'Iris-virginica': 'images/Iris-virginica.jpg'
    }

    # Convert species to an integer index
    try:
        species_index = int(float(species))  # Convert from string to float, then to int
        predicted_species = species_names[species_index]
        species_image = species_images[predicted_species]
        
    except (ValueError, IndexError):
        return "Invalid species prediction.", 400  # Handle possible errors

    return render_template('result.html', predicted_species=predicted_species, species_image=url_for('static', filename=species_image))



if __name__ == '__main__':
    app.run(debug=True)
