import warnings

import numpy as np
import pandas as pd
from keras.models import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)

PATH = './models/'


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
