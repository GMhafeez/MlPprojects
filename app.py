from flask import request, render_template, Flask
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData
from src.Pipeline.predict_pipeline import Predictpipeline

application = Flask(__name__)

app = application 

## route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch= request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            writing_score= request.form.get('writing_score'),
            reading_score= request.form.get('reading_score')
        )
        pred_df_data = data.get_data_as_frame()
        print(pred_df_data)

        predict_pipeline = Predictpipeline()
        results = predict_pipeline.predict_data(pred_df_data)

        return render_template('home.html', results=results[0])
    
if __name__ == '__main__':
    app.run(debug=True,port=5000)
    