import sys
import math
from pprint import pprint
from itertools import chain, combinations
from datetime import datetime
import random

import pandas as pd
import numpy as np
from sklearn import cross_validation, tree, svm, linear_model, preprocessing, \
    neighbors
import matplotlib.pyplot as plt

def rmsle(actual_values, predicted_values):
    '''
        Implementation of Root Mean Squared Logarithmic Error
        See https://www.kaggle.com/c/bike-sharing-demand/details/evaluation
    '''
    assert len(actual_values) == len(predicted_values), \
            "Both input paramaters should have the same length"

    # Depending on the regression method, the input paramaters can be either
    # a numpy.ndarray or a list, for the formet we need to convert it to a list
    if type(actual_values) is np.ndarray:
        actual_values = np.reshape(actual_values, len(actual_values))
    if type(predicted_values) is np.ndarray:
        predicted_values = np.reshape(predicted_values, len(predicted_values))

    total = 0
    for a, p in zip(actual_values, predicted_values):
        total += math.pow(math.log(p+1) - math.log(a+1), 2)

    return math.sqrt(total/len(actual_values))

def powerset(iterable):
    '''
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def pre_process_data(data, selected_columns):
    '''
        Does some pre-processing on the existing columns and only keeps
        columns present in [selected_columns].

        Returns a numpy array
    '''

    # Some 'magic' string to datatime function
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Since the hour of day is cyclical, e.g. 01:00 is equaly far from midnight
    # as 23:00 we need to represent this in a meaningful way. We use both sin
    # and cos, to make sure that 12:00 != 00:00 (which we cannot prevent if we only
    # use sin)
    data['hour_of_day'] = data['datetime'].apply(lambda i: i.hour)
    data['hour_of_day_sin'] = data['hour_of_day'].apply(lambda hour: math.sin(2*math.pi*hour/24))
    data['hour_of_day_cos'] = data['hour_of_day'].apply(lambda hour: math.cos(2*math.pi*hour/24))

    first_day = datetime.strptime('2011-01-01', "%Y-%m-%d").date()
    data['day_since_begin'] = data['datetime'].apply(lambda i: (i.date()-first_day).days)

    # Some variables have no numerical value, they are categorical. E.g. the weather
    # variable has numerical values, but they cannot be interpreted as such.
    # In other words value 2 is not two times as small as value 4.
    # A method to deal with this is one-hot-enconding, which splits the existing
    # variable in n variables, where n equals the number of possible values.
    # See
    for column in ['season', 'weather']:
        dummies = pd.get_dummies(data[column])
        # Concat actual column name with index
        new_column_names = [column + str(i) for i in dummies.columns]
        data[new_column_names] = dummies

    data = data[selected_columns]

    # Let scale the data set to have zero mean and unit variance, this makes sure variables with different orders of magnitude
    # are treated equally
    #data = preprocessing.scale(data)
    data = data.values

    return data

def main():
    data = pd.read_csv("data/train.csv")
    data = pd.DataFrame(data = data)

    features = ['day_since_begin', 'hour_of_day_cos', 'hour_of_day_sin', 'workingday',
        'temp', 'atemp', 'weather1', 'weather2', 'weather3', 'weather4', 'season1',
         'season2', 'season3', 'season4']

    train_data = pre_process_data(data, features)
    train_labels = data[['count']].values.astype(float)

    kf = cross_validation.KFold(len(data), n_folds=5, shuffle=False)

    scores = []
    for train_index, test_index in kf:
        # Train the model
        clf = tree.DecisionTreeRegressor()
        #clf = svm.SVR(kernel='rbf', C=1e3)
        #clf = linear_model.Ridge (alpha = 1.5)
        #clf = linear_model.Lasso(alpha = 0.1)
        #clf = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
        clf.fit(train_data[train_index],np.ravel(train_labels[train_index]))
        print clf
        # Test it
        predicted = clf.predict(train_data[test_index])

        # Some methods can predict negative values
        predicted = [p if p > 0 else 0 for p in predicted]

        df = pd.DataFrame({'datetime': data['datetime'].values[test_index],
            'true': np.ravel(train_labels[test_index]),'predicted': np.ravel(predicted)})
        index = random.randint(0,len(df))
        df[index:index+48].plot(x='datetime')
        plt.show()

        scores.append(rmsle(train_labels[test_index], predicted))

    # Print average cross-validation score
    avg = sum(scores) / len(scores)
    print "Average RMSLE:", avg

    # Train on all data
    clf.fit(train_data,np.ravel(train_labels))

    # Predict test data

    test_data = pd.read_csv("data/test.csv")
    test_data = pd.DataFrame(data = test_data)
    transformed_test_data = pre_process_data(test_data, features)

    # Predict all test data
    predicted = clf.predict(transformed_test_data)

    # Write the output to a csv file
    output = pd.DataFrame()
    output['datetime'] = test_data['datetime']
    output['count'] = predicted

    # Don't write row numbers, using index=False
    output.to_csv('data/predicted.csv', index=False)


if __name__ == "__main__":
    main()
