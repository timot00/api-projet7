import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from data_api import data_api
       

app = FastAPI()


pickle_in = open('LGBMClassifier.pkl', 'rb') 
model = pickle.load(pickle_in)


description = pd.read_csv("features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
data = pd.read_csv('input_data_model2.zip', index_col='SK_ID_CURR', encoding ='utf-8')

#data = pd.read_csv('input_data_model.csv.zip', index_col='SK_ID_CURR', encoding ='utf-8')
data = data.drop('Unnamed: 0', 1)
data = data.drop('index', 1)

target = data.iloc[:, -1:]

    
@app.get('/')

def index():

    return {'message': 'Page principale de l application de prédiction'}


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to the Credit Scoring API. We try to predict if you will be in default or not': f'{name}'}


@app.post('/predict')
def predict_creditstatus(data: data_api):
    data = data.dict()
    print(data)
    EXT_SOURCE_1 = data['EXT_SOURCE_1']
    EXT_SOURCE_2 = data['EXT_SOURCE_2']
    EXT_SOURCE_3 = data['EXT_SOURCE_3']
    

    prediction = classifier.predict_proba([[
                    EXT_SOURCE_1,
                    EXT_SOURCE_2,
                    EXT_SOURCE_3
                     ]])
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
    
    # uvicorn.run(app)
    
    
    
    
    