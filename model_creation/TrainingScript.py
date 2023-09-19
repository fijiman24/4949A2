import pickle
import warnings
from os import makedirs
from os import path

import numpy as np
import pandas as pd
import tensorflow
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action="ignore", category=FutureWarning)

PATH = './models/'


# Methods
def view_quick_stats(dataframe):
    """
    Generate quick views of data.
    """
    print("\n*** Show contents of the file.")
    print(dataframe.head(11))
    print("\n*** Show the description for all columns.")
    print(dataframe.info())
    print("\n*** Describe numeric values.")
    print(dataframe.describe())


def get_non_numeric_attributes(dataframe):
    """
    Create a list of non-numeric attributes of dataframe.
    :param dataframe: dataframe
    :return: list of non-numeric attributes
    """
    non_numeric_attributes = []

    for name, data_type in dataframe.dtypes.items():
        if data_type == 'object' or data_type == 'category':
            non_numeric_attributes.append(name)

    return non_numeric_attributes


def get_numeric_attributes(dataframe, predicted_variable):
    """
    Print all numeric attributes.
    :param dataframe: a dataframe
    :param predicted_variable: a String
    """
    non_numeric_variables = get_non_numeric_attributes(dataframe)

    for x in non_numeric_variables:
        print(x)
    print(len(non_numeric_variables))

    numeric_variables = []
    for x in dataframe:
        if x not in non_numeric_variables and x != predicted_variable:
            numeric_variables.append(x)

    for x in numeric_variables:
        print(x)
    print(len(numeric_variables))


def dummy_non_numeric_variables(dataframe):
    """
    Add dummy variables for non-numeric attributes in dataframe.
    :param dataframe: dataframe
    :return: dataframe with dummy variables appended
    """
    non_numeric_attributes = get_non_numeric_attributes(dataframe)
    temp_df = dataframe[non_numeric_attributes]  # Isolate columns
    dummy_df = pd.get_dummies(temp_df, columns=non_numeric_attributes)  # Get dummies
    dataframe = pd.concat(([dataframe, dummy_df]), axis=1)  # Join dummy df with original df

    return dataframe


def create_binned_variable(dataframe, variable_name, bins):
    """
    Create binned variables of a variable
    :param dataframe: a dataframe
    :param variable_name: a String
    :param bins: a list
    :return: dataframe with binned variables appended
    """
    bin_name = variable_name + 'Bin'
    dataframe[bin_name] = pd.cut(x=dataframe[variable_name], bins=bins)
    temp_df = dataframe[[bin_name]]
    dummy_df = pd.get_dummies(temp_df, columns=[bin_name])
    new_df = pd.concat((dataframe, dummy_df), axis=1)
    return new_df


def recursive_feature_elimination(X, y):
    significant_variables = []
    model = LogisticRegression()
    rfe = RFE(model)
    rfe = rfe.fit(X, y)

    for i in range(0, len(X.keys())):
        if rfe.support_[i]:
            significant_variables.append(X.keys()[i])

    return significant_variables


def forward_feature_selection(X, y):
    # f_regression is a scoring function to be used in a feature selection procedure
    # f_regression will compute the correlation between each regressor and the target
    ffs = f_regression(X, y)
    significant_variables = []

    for i in range(0, len(X.columns) - 1):
        if ffs[0][i] >= 3:
            significant_variables.append(X.columns[i])

    return significant_variables


def get_common_significant_variables(rfe_significant, ffs_significant):
    print("\nRFE:")
    print(rfe_significant)

    print("\nForward feature selection:")
    print(ffs_significant)

    common_significants = list(set(rfe_significant) & set(ffs_significant))

    print("\nCommon significant variables:")
    print(common_significants)

    return common_significants


def create_ann(numNeurons=5, numHiddenLayers=1, initializer='normal', activation='relu',
               learning_rate=0.005):
    model = Sequential()
    for i in range(numHiddenLayers):
        model.add(Dense(numNeurons, kernel_initializer=initializer, activation=activation,
                        input_shape=(n_features,)))
    # Output layer
    model.add(Dense(2, activation='softmax'))
    opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
    # Compile the model.
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def generateModels(numNeurons, numHiddenLayers, initializer, activation, learning_rate):
    # create directory for models
    if (not path.exists(PATH)):
        makedirs('./models')

    # fit and save models
    numModels = 5
    print("\nFitting models with training data.")
    for i in range(numModels):
        filename = PATH + 'model_' + str(i + 1) + '.h5'

        # fit model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001,
                           patience=50)
        model = create_ann(numNeurons, numHiddenLayers, initializer, activation, learning_rate)
        model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
                  validation_data=(X_val, y_val), callbacks=[es])

        # save model
        model.save(filename)
        print('>Saved %s' % filename)


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of models
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def getStackedData(models, inputX):
    stackXdf = None
    for model in models:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        singleModelPredDf = pd.DataFrame(np.row_stack(yhat))

        if stackXdf is None:
            stackXdf = singleModelPredDf
        else:
            numClasses = len(singleModelPredDf.keys())
            numStackXCols = len(stackXdf.keys())

            # Add new classification columns.
            for i in range(0, numClasses):
                stackXdf[numStackXCols + i] = stackXdf[i]
    return stackXdf


# Make predictions with the stacked model
def stacked_prediction(models, stackedModel, inputX):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # make a prediction
    yhat = stackedModel.predict(stackedX)
    return yhat


# fit a model based on the outputs from the ensemble models
def fit_stacked_model(models, inputX, inputy):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    save_model_as_pickle(model)
    return model


def save_model_as_pickle(model):
    # Save model as pickle
    filehandler = open(b"./models/stacked.pkl", "wb")
    pickle.dump(model, filehandler)


def display_boxplot(predictor, target):
    import matplotlib.pyplot as plt
    import seaborn as sns


    sns.set_style('whitegrid')
    sns.boxplot(y=predictor, x=target, data=df)
    sns.stripplot(y=predictor, x=target, data=df)
    plt.show()

if __name__ == '__main__':
    # Import Data
    CSV_DATA = "train.csv"
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

    # Enable the display of all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Delete ID from df
    del df['id']

    # Create dummy variables for non-numeric variables
    df = dummy_non_numeric_variables(df)

    # Delete duplicates of target variable
    del df['diagnosis']
    del df['diagnosis_B']

    # Separate target variable
    X = df.copy()
    predicted_variable = "diagnosis_M"
    del X[predicted_variable]
    y = df[predicted_variable]

    # # Builds all models with significant variables.
    significant_rfe = recursive_feature_elimination(X, y)
    significant_ffs = forward_feature_selection(X, y)
    significant_common = get_common_significant_variables(significant_rfe,
                                                          significant_ffs)
    X = df[significant_common]

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.values.reshape(X.shape[ROW_DIM],
                                       X.shape[COL_DIM])

    # Convert DataFrame columns to vertical columns of target variables values.
    y_arrayReshaped = y.values.reshape(y.shape[ROW_DIM], 1)

    X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.3,
                                                        random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    n_features = X_train.shape[1]

    # # # At least optimizes the number of nodes, learning rate, number of layers, activation function and kernel initializer for the network model
    # # # Grid Search takes 2 hours, best model has these params:
    # # # {'activation': 'linear', 'initializer': 'normal', 'learning_rate': 0.01, 'numHiddenLayers': 2, 'numNeurons': 25}
    # # params = {'activation': ['softmax', 'relu', 'tanh', 'sigmoid', 'linear'],
    # #           'numNeurons': [10, 15, 20, 25, 30, 35],
    # #           'numHiddenLayers': [1, 2, 3],
    # #           'initializer': ['uniform', 'normal', 'zero', 'glorot_normal', 'he_normal'],
    # #           'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
    # #           }
    # #
    # # model = KerasRegressor(build_fn=create_ann, epochs=100,
    # #                        batch_size=9, verbose=1)
    # #
    # # # # Saves your model binary files with early stopping
    # # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
    # # mc = ModelCheckpoint('best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # #
    # # # Fit the model.
    # # grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3)
    # # grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0,
    # #                        callbacks=[es, mc])
    # #
    # # # load the saved model
    # # model = load_model('best.h5')
    # # ############################################

    generateModels(activation='linear', initializer='normal', learning_rate=0.01, numHiddenLayers=2,
                   numNeurons=25)

    # load all models
    numModels = 5
    models = load_all_models(numModels)
    print('Loaded %d models' % len(models))

    print("\nEvaluating single models with test data.")
    for model in models:
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Model Accuracy: %.3f' % acc)

    # fit stacked model using the ensemble
    # Stacked model build with LogisticRegression.
    # y for LogisticRegression is not one-hot encoded.
    print("\nFitting stacked model with test data.")
    stackedModel = fit_stacked_model(models, X_val, y_val)

    # evaluate model on test set
    print("\nEvaluating stacked model with test data.")
    yhat = stacked_prediction(models, stackedModel, X_test)
    acc = accuracy_score(y_test, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
