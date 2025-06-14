from flask import Flask, render_template, request
import os
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input from form and convert comma-separated to float list
            features_str = request.form['features']
            features = [float(x.strip()) for x in features_str.split(',')]
            final = [np.array(features)]
            prediction = model.predict(final)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}' if prediction else '')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
