from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('C:/Users/DELL/Desktop/Titanic Classification/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    pclass = int(request.form['Pclass'])
    sex = int(request.form['Sex'])
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])
    embarked = int(request.form['Embarked'])

    # Create feature array for prediction
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    # Predict survival
    prediction = model.predict(features)
    result = 'Survived' if prediction[0] == 1 else 'Not Survived'

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
