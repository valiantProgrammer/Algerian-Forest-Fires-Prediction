import pickle
from flask import Flask,request, jsonify, render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

##Import Ridge Regression and Standard Scaler pickle
ridge=pickle.load(open('models/ridge-reg.pkl', 'rb'))
scaler=pickle.load(open('models/scaler.pkl', 'rb'))

application = Flask(__name__)
app=application
@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/predictdata',methods=['GET', 'POST'])
def predict():
    results = None
    if request.method == 'POST':
        Temperature = int(request.form.get('temperature'))
        Relative_humidity = int(request.form.get('rh'))
        wind_speed = int(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        classes = int(request.form.get('class'))
        region = int(request.form.get('region'))
        
        new_data_scaled = scaler.transform([[Temperature,Relative_humidity,wind_speed,rain,ffmc,dmc,isi,classes,region]])
        result = ridge.predict(new_data_scaled)
        results = result[0]
        
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)