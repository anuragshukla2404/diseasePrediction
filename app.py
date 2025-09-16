from flask import Flask, render_template, request

import pickle
import numpy as np

app = Flask(__name__)

# Load your model and label encoder
model = pickle.load(open('classifier.pkl', 'rb'))
category = pickle.load(open('diseases.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features)

        predicted_label = category[int(prediction[0])]

        return render_template('index.html', prediction_text=f'Result: {predicted_label}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
