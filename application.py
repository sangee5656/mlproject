from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
         # Get the value of 'reading_score' from the form data
        reading_score_str = request.form.get('reading_score')
        
        # Check if 'reading_score' is not None and not empty
        if reading_score_str is not None and reading_score_str.strip():
            try:
                # Convert the value of 'reading_score' to an integer
                reading_score = int(reading_score_str)
            except ValueError:
                # Handle the case when 'reading_score' cannot be converted to an integer
                return "Error: 'reading_score' must be a valid integer", 400
        else:
            # Handle the case when 'reading_score' is not provided or empty
            return "Error: 'reading_score' is missing or empty", 400
        
        # Continue with the rest of the function logic using 'reading_score'
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=reading_score,  # Use the validated '
            writing_score=int(request.form.get('writing_score'))


        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        #print("Before Prediction")

        predict_pipeline=PredictPipeline()
        #print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        #print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


## http://127.0.0.1:5000/predictdata   check here 