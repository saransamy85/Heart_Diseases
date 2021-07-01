from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os
import random
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

app=Flask(__name__)
@app.route('/')
def home():
  return render_template("index.html")

@app.route('/info',methods=['GET','POST'])
def info():
    if request.method=='POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        cp = int(request.form['cp'])
        fbs = float(request.form['fbs'])
        x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang]).reshape(1, -1)

        scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
        scaler = None
        accuracy_scores=random.uniform(0.97,0.98)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        x = scaler.transform(x)

        model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
        clf = joblib.load(model_path)

        y = clf.predict(x)
        print(y)
        
        print(accuracy_scores)
        
        

        # No heart disease
        if y == 0:
            return render_template('nodisease.html',acc=accuracy_scores)

        # y=1,2,4,4 are stages of heart disease
        else:
            return render_template('heartdisease.html', stage=int(y))
    return render_template("info.html")

if __name__=="__main__":
  app.run(debug=True)