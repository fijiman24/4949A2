# Tutorial source: https://towardsdatascience.com/how-to-easily-build-your-first-machine-learning-web-app-in-python-c3d6c0f0a01c
from flask import Flask, render_template, request
import pickle
from TrainingScript import load_all_models, getStackedData

app = Flask(__name__)

# load all ANN models
numModels = 5
models = load_all_models(numModels)
print('Loaded %d models' % len(models))

# load stacked model
file = open("./models/stacked.pkl", 'rb')
stacked_model = pickle.load(file)
print('Loaded stacked model')


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    compactness_worst = float(request.form['compactness_worst'])
    concavity_worst = float(request.form['concavity_worst'])
    radius_worst = float(request.form['radius_worst'])
    radius_se = float(request.form['radius_se'])
    radius_mean = float(request.form['radius_mean'])
    symmetry_worst = float(request.form['symmetry_worst'])
    concavity_mean = float(request.form['concavity_mean'])
    concavity_se = float(request.form['concavity_se'])
    compactness_se = float(request.form['compactness_se'])
    concave_points_mean = float(request.form['concave points_mean'])
    concave_points_worst = float(request.form['concave points_worst'])
    perimeter_worst = float(request.form['perimeter_worst'])
    perimeter_se = float(request.form['perimeter_se'])
    compactness_mean = float(request.form['compactness_mean'])

    X = [[compactness_worst, concavity_worst, radius_worst, radius_se, radius_mean, symmetry_worst,
          concavity_mean, concavity_se, compactness_se, concave_points_mean, concave_points_worst,
          perimeter_worst, perimeter_se, compactness_mean]]

    stackedX = getStackedData(models, X)
    prediction = stacked_model.predict(stackedX)
    output = 'benign' if prediction[0] == 0 else 'malignant'
    return render_template('index.html', prediction_text=f'The tumor is {output}.')


if __name__ == "__main__":
    app.run()
