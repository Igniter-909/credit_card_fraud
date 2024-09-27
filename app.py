from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app =application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictionPage', methods=['GET','POST'])
def  predict_datapoint():
    if request.method=='GET':
        return render_template('predictionPage.html')
    else:
        data = CustomData(
            Time = float(request.form.get('time')),
            V1 = float(request.form.get('v1')),
            V2 = float(request.form.get('v2')),
            V3 = float(request.form.get('v3')),
            V4 = float(request.form.get('v4')),
            V5 = float(request.form.get('v5')),
            V6 = float(request.form.get('v6')),
            V7 = float(request.form.get('v7')),
            V8 = float(request.form.get('v8')),
            V9 = float(request.form.get('v9')),
            V10 = float(request.form.get('v10')),
            V11 = float(request.form.get('v11')),
            V12 = float(request.form.get('v12')),
            V13 = float(request.form.get('v13')),
            V14 = float(request.form.get('v14')),
            V15 = float(request.form.get('v15')),
            V16 = float(request.form.get('v16')),
            V17 = float(request.form.get('v17')),
            V18 = float(request.form.get('v18')),
            V19 = float(request.form.get('v19')),
            V20 = float(request.form.get('v20')),
            V21 = float(request.form.get('v21')),
            V22 = float(request.form.get('v22')),
            V23 = float(request.form.get('v23')),
            V24 = float(request.form.get('v24')),
            V25 = float(request.form.get('v25')),
            V26 = float(request.form.get('v26')),
            V27 = float(request.form.get('v27')),
            V28 = float(request.form.get('v28')),
            Amount = float(request.form.get('amount'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predictpipeline = PredictPipeline()
        results = predictpipeline.predicts(pred_df)
        return render_template('predictionPage.html',results=results)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
        
