from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from e1_create_dataset import create_classification_dataset, create_regression_dataset
from e3_metrics import classification_scores, regression_scores

import numpy as np

# Recommended reading: https://scikit-learn.org/stable/modules/cross_validation.html

# For time series data, see: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data

def holdout():
    # Create a dataset and a model
    _, X, y = create_classification_dataset()
    model = DecisionTreeClassifier()

    # Train using a holdout methodology:
    # 1. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 2. Train the model using the training set
    model.fit(X_train, y_train)

    # 3. Make predictions using the trained model on the test set
    preds = model.predict(X_test)

    # 4. Evaluate the model
    print('\nHoldout classification training score:', classification_scores(y_train, model.predict(X_train))['accuracy'])
    print('Holdout classification testing score:', classification_scores(y_test, preds)['accuracy'])


def cross_validation():
    # Create a dataset
    _, X, y = create_classification_dataset()

    # Train and evaluate the model
    # See other scoring metrics here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    cv_results = cross_validate(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy', return_train_score=True)
    print('\nCV classification training score:', cv_results['train_score'].mean())
    print('CV classification testing scores:', cv_results['test_score'])
    print('CV classification testing scores (averaged):', cv_results['test_score'].mean())


def time_series_regression_holdout():
    _, X, y = create_regression_dataset()
    model = DecisionTreeRegressor()

    # Train using a holdout methododlogy
    # 1. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 2. Train the model using the training set
    model.fit(X_train, y_train)

    # 3. Make predictions using the trained model on the test set
    preds = model.predict(X_test)

    # 4. Evaluate the model
    print('\nHoldout regression training RMSE:', regression_scores(y_train, model.predict(X_train))['RMSE'])
    print('Holdout regression testing RMSE:', regression_scores(y_test, preds)['RMSE'])


def time_series_regression_cross_validation():
    # Create a dataset and a model
    _, X, y = create_regression_dataset()
    model = DecisionTreeRegressor()

    # Create time-series split
    tss = TimeSeriesSplit(n_splits=3)

    # Creating two arrays to store the performance scores
    train_scores = []
    test_scores = []

    # Split train test sets and training on them
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fitting the model
        model.fit(X_train, y_train)

        # Predicting results for both training and test set
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Computing scores and saving them
        train_scores.append(regression_scores(y_train, train_preds)['RMSE'])
        test_scores.append(regression_scores(y_test, test_preds)['RMSE'])

    # Adding the scores to a scores dictionary - like in cross_validate -> converted to NumPy arrays to be consistent with cross_validate
    scores = {'train_score':np.asarray(train_scores), 'test_score':np.asarray(test_scores)}

    # Evaluate the model
    print('\nTime series cross validation regression training RMSE (averaged):', scores['train_score'].mean())
    print('Time series cross validation regression testing RMSE:', scores['test_score'])
    print('Time series cross validation regression testing RMSE (averaged):', scores['test_score'].mean())


if __name__ == '__main__':
    holdout()
    cross_validation()

    # Which scores are better?
    # Why do the cross-validation scores vary?
    # How do these scores compare to the previous exercise?

    time_series_regression_holdout()
    time_series_regression_cross_validation()