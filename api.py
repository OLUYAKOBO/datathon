from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

#import pickle files

model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

#create app
app = FastAPI()

#define the class for the features

class PredictRequest(BaseModel):
    #numerical columns
    study_hours: float
    acad_performance: float
    stud_percent_pass: float
    stud_performance_impact: float
    number_of_children: int

    #categorical columns 

    gender: str
    parental_involvement : str
    fac_standard : str
    activity_name : str
    participation_level: str
    achievement: str
    acad_level: str
    job_satisfaction: str
    salary_satisfaction : str
    marital_status: str
    edu_background : str
    occupation : str
    income_level : str


@app.get("/")
def read_root():
    return {"Message": "Welcome to the student exam outcome API"}


#define the prediction endpoint

@app.post("/predict")
def predict(request: PredictRequest):
    #numerical columns
    num_cols = np.array([[
        request.study_hours,
        request.acad_performance,
        request.stud_percent_pass,
        request.stud_performance_impact,
        request.number_of_children
    ]])

    #categorical values

    cat_cols = np.array([[

    request.gender,
    request.parental_involvement,
    request.fac_standard,
    request.activity_name,
    request.participation_level,
    request.achievement,
    request.acad_level,
    request.job_satisfaction,
    request.salary_satisfaction,
    request.marital_status,
    request.edu_background,
    request.occupation,
    request.income_level
    ]])

    #encode categorical values
    encoded_values = encoder.transform(cat_cols).toarray()

    #scale numerical features
    scaled_features = scaler.transform(num_cols)

    #combine the encoded and scaled features

    features = np.concatenate([scaled_features,encoded_values],
                              axis = 1)
    
    #make predictions
    prediction = model.predict(features)[0]

    result = "This Student is likely to pass the exam" if prediction == 1 else "This Student is likely to fail the exam"

    return {'Exam Outcome': result}



