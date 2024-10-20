import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 


from flask import Flask ,request,jsonify,render_template


ridge_model= pickle.load(open("models/ridge.pkl",'rb'))


standard_scaler= pickle.load(open("models/scaler.pkl",'rb'))


application= Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature= int(request.form.get("Temperature"))
        RH= int(request.form.get('RH'))
        Ws= int(request.form.get('Ws'))
        Rain= float(request.form.get('Rain'))
        FFMC= float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI= float(request.form.get('ISI'))
        Classes= int(request.form.get('Classes'))
        Region= int(request.form.get('Region'))
        
        # new_data_scaled=standard_scaler.transform([[Temperature, RH, Ws, Rain,FFMC,DMC,ISI,Classes,Region]])
        # result=ridge_model.predict(new_data_scaled)
        
        
        # return render_template("home.html",results=result[0])
        
        # Create a DataFrame for scaling
        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], 
                                    columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])

        # Print input data for debugging
        print("Input Data for Scaling:")
        print(input_data)

        # Scale the input data
        new_data_scaled = standard_scaler.transform(input_data)

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", results=result[0])
        
    else :
        return render_template("home.html")




if __name__=="__main__":
    app.run(host="0.0.0.0")

