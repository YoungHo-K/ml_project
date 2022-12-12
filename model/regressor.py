import time
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


class EvaluationMetrics:
    def __init__(self):
        self.learning_time_list = list()

        self.train_mean_squared_error_list = list()
        self.train_mean_absolute_error_list = list()

        self.test_mean_squared_error_list = list()
        self.test_mean_absolute_error_list = list()

    def calculate(self, learning_time, y_train_true, y_train_pred, y_test_true, y_test_pred):
        self.learning_time_list.append(learning_time)

        self.train_mean_squared_error_list.append(mean_squared_error(y_train_true, y_train_pred))
        self.train_mean_absolute_error_list.append(mean_absolute_error(y_train_true, y_train_pred))

        self.test_mean_squared_error_list.append(mean_squared_error(y_test_true, y_test_pred))
        self.test_mean_absolute_error_list.append(mean_absolute_error(y_test_true, y_test_pred))

    def get_result(self):
        np.set_printoptions(suppress=True)

        msg = "-------------- Evaluation Result --------------"
        msg += f"\n Number of evaluation:            {len(self.learning_time_list)}"
        msg += f"\n Learning time:                   {np.round(np.mean(self.learning_time_list), 3):.03f}"
        msg += f"\n --"
        msg += f"\n Train mean squared error:        {np.round(np.mean(self.train_mean_squared_error_list), 3):.03f}"
        msg += f"\n Train mean absolute error:       {np.round(np.mean(self.train_mean_absolute_error_list), 3):.03f}"
        msg += f"\n --"
        msg += f"\n Test mean squared error:         {np.round(np.mean(self.test_mean_squared_error_list), 3):.03f}"
        msg += f"\n Test mean absolute error:        {np.round(np.mean(self.test_mean_absolute_error_list), 3):.03f}"
        msg += "\n\n"

        return msg


class Regressor:
    def __init__(self, regressor=None, cv=5):
        if regressor is None:
            raise Exception("[ERROR] Invalid regressor.")

        self.regressor = regressor
        self.cv = cv

    def evaluate(self, X=None, y=None):
        if (X is None) or (y is None):
            raise Exception("[ERROR] Invalid parameters.")

        evaluation_metrics = EvaluationMetrics()

        ensemble_list = list()

        kfold = KFold(n_splits=self.cv, shuffle=True)
        for train_indexes, test_indexes in kfold.split(X):
            X_train_data = X[train_indexes]
            X_test_data = X[test_indexes]
            y_train_data = y[train_indexes]
            y_test_data = y[test_indexes]

            regressor = clone(self.regressor)

            start_time = time.time()
            regressor.fit(X_train_data, y_train_data)
            elapsed = time.time() - start_time

            y_train_pred = regressor.predict(X_train_data)
            y_test_pred = regressor.predict(X_test_data)

            evaluation_metrics.calculate(elapsed, y_train_data, y_train_pred, y_test_data, y_test_pred)

            ensemble_list.append(regressor)

        print(evaluation_metrics.get_result())

        return ensemble_list




