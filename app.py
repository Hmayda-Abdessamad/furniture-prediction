import pickle

import numpy as np
from flask import Flask, request, render_template
from joblib import load
app = Flask(__name__)

# Load the model using joblib
model = load('model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    '''
    pour l'affichage sur html
    '''

    features = request.form.to_dict()
    features = list(features.values())
    features = list(map(int, features))
    print(features)
    final_features = np.array(features).reshape(1,6)
    prediction = model.predict(final_features)

    #select = request.form.get('category')
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Furniture prediction price is : $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)