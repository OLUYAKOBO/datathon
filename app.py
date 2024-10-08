from flask import Flask, render_template,jsonify,request

import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

#load pickle files
model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route("/")

def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])

def predict():

    #numerical columns
    study_hours = float(request.form['study_hours'])
    acad_performance = float(request.form['acad_performance'])
    stud_percent_pass = float(request.form['stud_percent_pass'])
    stud_performance_impact = float(request.form['stud_performance_impact'])
    number_of_children = float(request.form['number_of_children'])

    #categorical columns
    gender = request.form['gender']
    parental_involvement = request.form['parental_involvement']
    fac_standard = request.form['fac_standard']
    activity_name = request.form['activity_name']
    participation_level = request.form['participation_level']
    achievement = request.form['achievement']
    acad_level = request.form['acad_level']
    job_satisfaction = request.form['job_satisfaction']
    salary_satisfaction = request.form['salary_satisfaction']
    marital_status = request.form['marital_status']
    edu_background = request.form['edu_background']
    occupation = request.form['occupation']
    income_level = request.form['income_level']


    #what we have done here is that, we only encoded the categorical features, we did not scale them
    encoded_values = encoder.transform([[gender,parental_involvement,
                                         fac_standard,activity_name,
                                         participation_level,
                                         achievement,
                                         acad_level,
                                         job_satisfaction,
                                         salary_satisfaction,
                                         marital_status,
                                         edu_background,
                                         occupation,
                                         income_level]]).toarray()
    
    num_features = np.array([[study_hours,acad_performance,stud_percent_pass,
                              stud_performance_impact,
                              number_of_children]])
    
    #we scaled the numerical features here
    num_features = scaler.transform(num_features)
    # we combined the scaled and encoded together here
    features = np.concatenate([num_features,encoded_values],axis = 1)


    prediction = model.predict(features)[0]

    if prediction == 0:
        prediction_text = "This student is likely to fail the exam"
    else:
        prediction_text = "There is a high chance that this student passes the exam"
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)





