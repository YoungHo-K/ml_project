"""
Reference: https://github.com/amiratag/DataShapley
"""

import tqdm
import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score


class TruncatedMonteCarloShapley:
    def __init__(self, classifier=None, tolerance=None, stopping_criteria=0.01, init_size=16, number_of_data_to_check=100, number_of_iteration_for_tolerance=100, metric="accuracy"):
        if classifier is None:
            raise Exception("[ERROR] Invalid classifier.")

        self.classifier = classifier
        self.tolerance = tolerance
        self.stopping_criteria = stopping_criteria
        self.init_size = init_size
        self.number_of_data_to_check = number_of_data_to_check
        self.number_of_iteration_for_tolerance = number_of_iteration_for_tolerance
        self.metric = metric

    def get_shapley_values(self, X_train_data, y_train_data, X_test_data, y_test_data):
        shapley_values = self._calculate_shapley(X_train_data, y_train_data, X_test_data, y_test_data)

        return shapley_values

    def _calculate_shapley(self, X_train_data, y_train_data, X_test_data, y_test_data):
        shapley_values = np.zeros((1, len(X_train_data)))

        number_of_iterations = 0
        while True:
            number_of_iterations += 1

            error = self._get_error(shapley_values)
            if error < self.stopping_criteria:
                break

            print(f"[INFO] #{number_of_iterations} Iteration : ERROR - {error}, STOPPING CRITERION - {self.stopping_criteria}")
            print(f"  -- Get tolerance performance.")

            mean_score, tolerance = self._get_tolerance_performance(X_train_data, y_train_data, X_test_data, y_test_data, iteration=self.number_of_iteration_for_tolerance)
            if self.tolerance is not None:
                tolerance = self.tolerance

            print(f"  -- Start iteration to calculate marginal contribution.")
            marginal_contribution = self._iteration(X_train_data, y_train_data, X_test_data, y_test_data, mean_score, shapley_values[-1], tolerance, number_of_iterations)
            shapley_values = np.concatenate([shapley_values, marginal_contribution.reshape(1, -1)])

        return shapley_values[-1]

    def _iteration(self, X_train_data, y_train_data, X_test_data, y_test_data, mean_score, marginal_contribution, tolerance, number_of_iteration):
        batch_indexes = self._get_batch_data_indexes(y_train_data)

        X_train_batch_data = X_train_data[batch_indexes]
        y_train_batch_data = y_train_data[batch_indexes]

        old_score = self._init_performance_score(X_train_batch_data, y_train_batch_data, X_test_data, y_test_data)

        train_indexes = np.random.permutation(len(X_train_data))
        for index in tqdm.tqdm(train_indexes):
            if index in batch_indexes:
                marginal_contribution[index] = ((number_of_iteration - 1) * marginal_contribution[index]) / number_of_iteration

                continue

            if np.abs(mean_score - old_score) > tolerance:
                X_train_batch_data = np.concatenate([X_train_batch_data, X_train_data[index: index + 1]])
                y_train_batch_data = np.concatenate([y_train_batch_data, y_train_data[index: index + 1]])

                self.classifier = clone(self.classifier)
                self.classifier.fit(X_train_batch_data, y_train_batch_data)

                new_score = self._get_performance(X_test_data, y_test_data)

            else:
                new_score = old_score

            marginal_contribution[index] = ((number_of_iteration - 1) * marginal_contribution[index]) / number_of_iteration + (new_score - old_score) / number_of_iteration

        return marginal_contribution

    def _get_batch_data_indexes(self, y_data):
        unique_labels = np.unique(y_data)

        batch_indexes = list()
        for label in unique_labels:
            indexes_in_class = np.random.permutation(np.where(y_data == label)[0])[: self.init_size]

            batch_indexes.extend(list(indexes_in_class))

        return batch_indexes

    def _get_performance(self, X_data, y_data):
        y_pred = self.classifier.predict(X_data)

        if self.metric == "accuracy":
            return accuracy_score(y_data, y_pred)

        if self.metric == "f1-score":
            return f1_score(y_data, y_pred)

        raise Exception("[ERROR] Invalid metric.")

    def _get_error(self, shapley_values):
        if len(shapley_values) < self.number_of_data_to_check:
            return 1.0

        indexes = np.arange(1, len(shapley_values) + 1).reshape(-1 ,1)
        shapley_values = np.cumsum(shapley_values, 0) / indexes

        errors = np.abs(shapley_values[-self.number_of_data_to_check: ] - shapley_values[-1]) / (np.abs(shapley_values[-1]) + 1e-9)
        errors = np.mean(errors, -1)

        return np.max(errors)

    def _get_tolerance_performance(self, X_train_data, y_train_data, X_test_data, y_test_data, iteration=100):
        score_list = list()

        self.classifier = clone(self.classifier)
        self.classifier.fit(X_train_data, y_train_data)

        for _ in range(iteration):
            bagging_indexes = np.random.choice(len(X_test_data), len(X_test_data))
            score = self._get_performance(X_test_data[bagging_indexes], y_test_data[bagging_indexes])

            score_list.append(score)

        tolerance = np.std(score_list)
        mean_score = np.std(score_list)

        return mean_score, tolerance

    def _init_performance_score(self, X_train_data, y_train_data, X_test_data, y_test_data):
        self.classifier = clone(self.classifier)
        self.classifier.fit(X_train_data, y_train_data)

        score = self._get_performance(X_test_data, y_test_data)

        return score
