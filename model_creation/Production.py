import pickle

import pandas as pd

from TrainingScript import load_all_models, getStackedData

if __name__ == '__main__':
    CSV_DATA = "test.csv"
    df = pd.read_csv(CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                     names=('id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                            'smoothness_mean', 'compactness_mean', 'concavity_mean',
                            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                            'fractal_dimension_se', 'radius_worst', 'texture_worst',
                            'perimeter_worst',
                            'area_worst', 'smoothness_worst', 'compactness_worst',
                            'concavity_worst',
                            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
                            'diagnosis'))

    # Delete ID and target from df
    del df['id']
    del df['diagnosis']

    # Need to ensure order of variables matches those chosen in training script, which changes with each run
    predictor_variables = ['compactness_se',
                           'compactness_worst',
                           'concavity_mean',
                           'radius_worst',
                           'concave points_worst',
                           'radius_mean',
                           'concavity_worst',
                           'symmetry_worst',
                           'compactness_mean',
                           'perimeter_se',
                           'radius_se',
                           'concavity_se',
                           'concave points_mean',
                           'perimeter_worst']

    X = df[predictor_variables]

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.values.reshape(X.shape[ROW_DIM],
                                       X.shape[COL_DIM])

    # load all models
    numModels = 5
    models = load_all_models(numModels)
    print('Loaded %d models' % len(models))

    # Load stacked model.
    file = open("./models/stacked.pkl", 'rb')
    stacked_model = pickle.load(file)

    stackedX = getStackedData(models, x_arrayReshaped)
    predictions = stacked_model.predict(stackedX)

    print(predictions)

    # Store predictions in a dataframe
    dfPredictions = pd.DataFrame(predictions)
    dfPredictions = dfPredictions.rename({0: 'diagnosis_M'}, axis=1)
    listPredictions = []

    for i in range(0, len(predictions)):
        prediction = int(dfPredictions['diagnosis_M'][i])
        listPredictions.append(prediction)
    dfPredictions['diagnosis_M'] = listPredictions

    dfPredictions.to_csv('predictions.csv', index=False)
