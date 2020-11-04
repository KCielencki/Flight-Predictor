from flask import Flask, render_template, request,  jsonify
import pickle
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictor")
def predictor():
    return render_template("predictor.html")

@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html")

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/getPrice/', methods=['POST'])
def getPrice():
    content = request.json["data"]
    origin_airport = content["origin"]
    destination_airport = content["dest"]
    my_prod_model = pickle.load(open('static/model/ml_rf_fp.sav', 'rb'))

    df = pd.read_csv("machineLearningData.csv")
    or_lat = -df.loc[df.Origin==origin_airport,'Origin_Lat'].values[0]
    or_long = df.loc[df.Origin==origin_airport,'Origin_Long'].values[0]
    de_lat = -df.loc[df.Origin==destination_airport,'Origin_Lat'].values[0]
    de_long = df.loc[df.Origin==destination_airport,'Origin_Long'].values[0]

    #Convert to miles
    km = haversine_distance(or_lat, or_long, de_lat, de_long)
    mi = km * 0.62137
    
    #Observations
    number_coupons = 1
    miles = mi
    origin_lat = or_lat
    origin_long = or_long
    destination_lat = de_lat
    destination_long = de_long

    new_obs = [[number_coupons, miles, origin_lat, origin_long, destination_lat, destination_long]]

    fare = my_prod_model.predict(new_obs)

    return jsonify(list(map('${:.2f}'.format,fare)))

#Distance Calculator
def haversine_distance(or_lat, or_long, de_lat, de_long):
    r = 6371
    phi1 = np.radians(or_lat)
    phi2 = np.radians(de_lat)
    delta_phi = np.radians(de_lat - or_lat)
    delta_lambda = np.radians(de_long - or_long)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

if __name__ == "__main__":
    app.run(debug=True)
