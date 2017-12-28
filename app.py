from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form

        new_vector = []
        new_vector.append(0.0)
        try:
            new_vector.append(result['CrScore'])
        except:
            pass
        try:
            new_vector.append(result['Geography'])
        except:
            pass
        try:
            new_vector.append(result['Gender'])
        except:
            pass
        try:
            new_vector.append(result['Age'])
        except:
            pass
        try:
            new_vector.append(result['Tenure'])
        except:
            pass
        try:
            new_vector.append(result['Balance'])
        except:
            pass
        try:
            new_vector.append(result['NumofProducts'])
        except:
            pass
        try:
            new_vector.append(result['HasCrCard'])
        except:
            pass
        try:
            new_vector.append(result['IsActiveMember'])
        except:
            pass
        try:
            new_vector.append(result['EstimatedSalary'])
        except:
            pass
        
        pkl_file = open('clf.pkl', 'rb')
        logmodel = pickle.load(pkl_file)
        sc_file = open('sc.pkl', 'rb')
        sc = pickle.load(sc_file)
        print sc.data_max_
        new_vector2 = sc.transform(new_vector)
        prediction = logmodel.predict(new_vector2)
        
        return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)