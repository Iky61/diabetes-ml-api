# import library
import imp
from flask import Flask, request
from urllib import response
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from jcopml.tuning import grid_search_params as gsp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# read x_train data
x_train = pd.read_csv('x_train.csv')

# read y_train data
y_train = pd.read_csv('y_train.csv')

# create numeric pipeline
num_pipeline = Pipeline([
    ('scaling', StandardScaler())
])

# create processor columns transformer
processor = ColumnTransformer([
    ('numeric', num_pipeline, x_train.columns)
])

# create proces & metods/algoritm pipeline
pipeline = Pipeline([
    ('proces', processor),
    ('algo', SVC())
])

# create model using best params
model = GridSearchCV(pipeline, param_grid=gsp.svm_params)
model.fit(x_train, y_train)

class Test(Resource):
    def post(self):
        age = request.json['age']
        pregnancies = request.json['pregnancies']
        glucose = request.json['glucose']
        bloodpressure = request.json['bloodpressure']
        skinthickness = request.json['skinthickness']
        insulin = request.json['insulin']
        bmi = request.json['bmi']

        # create x_test data
        x_test = x_train.head(1)

        # input x_test data
        x_test['Pregnancies'] = pregnancies
        x_test['Glucose'] = glucose
        x_test['BloodPressure'] = bloodpressure
        x_test['SkinThickness'] = skinthickness
        x_test['Insulin'] = insulin
        x_test['BMI'] = bmi
        x_test['Age'] = age

        y_pred = model.predict(x_test)
        return {'value':str(y_pred[0])}

# inisiasi objeck flask
app = Flask(__name__)

# inisasi object flask_restful
api = Api(app)

# inisiasi object flask_cors
CORS(app)

# setup resource
api.add_resource(Test, "/api/predict", methods=["GET", "POST"])

if __name__ == "__main__":
    app.run()