"""
Reference : https://github.com/AI-secure/KNN-PVLDB
"""

import tqdm
import numpy as np


class KNNShapley:
    def __init__(self, number_of_neighbors=1):
        self.number_of_neighbors = number_of_neighbors

    def get_shapley_values(self, X_train_data, X_test_data,  y_train_data, y_test_data):
        print("[INFO] Get distance (X_train_data, X_test_data).")
        sorted_indexes = self._get_distance(X_train_data, X_test_data)

        print("[INFO] Calculate shapley values.")
        shapley_values = self._calculate_shapley(X_train_data, y_train_data, y_test_data, sorted_indexes)

        return shapley_values

    def _calculate_shapley(self, X_train_data, y_train_data, y_test_data, sorted_indexes):
        number_of_train_data = len(X_train_data)
        number_of_test_data = len(sorted_indexes)

        shapley_values = np.zeros((number_of_test_data, number_of_train_data))
        for index_of_test_data in tqdm.tqdm(range(0, number_of_test_data)):
            index_of_last_data = sorted_indexes[index_of_test_data, -1]
            shapley_values[index_of_test_data, index_of_last_data] = (y_test_data[index_of_test_data] == y_train_data[index_of_last_data]) / number_of_train_data

            for index_of_train_data in range(number_of_train_data - 2, -1, -1):
                index_of_selected_data = sorted_indexes[index_of_test_data, index_of_train_data]
                index_of_next_selected_data = sorted_indexes[index_of_test_data, index_of_train_data + 1]

                shapley_values[index_of_test_data, index_of_selected_data] = \
                    shapley_values[index_of_test_data, index_of_next_selected_data] + \
                    ((int(y_train_data[index_of_selected_data] == y_test_data[index_of_test_data]) -
                      int(y_train_data[index_of_next_selected_data] == y_test_data[index_of_test_data])) / self.number_of_neighbors) * \
                        (min([self.number_of_neighbors, index_of_train_data + 1]) / (index_of_train_data + 1))

        shapley_values = np.mean(shapley_values, axis=0)

        return shapley_values

    @staticmethod
    def _get_distance(X_train_data, X_test_data):
        number_of_train_data = len(X_train_data)
        number_of_test_data = len(X_test_data)

        distance_array = np.zeros((number_of_test_data, number_of_train_data))
        for index_of_test_data in range(0, number_of_test_data):
            distances = np.zeros(number_of_train_data)

            for index_of_train_data in range(0, number_of_train_data):
                distances[index_of_train_data] = np.linalg.norm(X_train_data[index_of_train_data, :] - X_test_data[index_of_test_data, :], 2)

            distance_array[index_of_test_data, :] = np.argsort(distances)

        distance_array = distance_array.astype(int)

        return distance_array



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    shapley = KNNShapley()
    shapley_values = shapley.get_shapley_values(X_train, X_test, y_train, y_test)

    print(shapley_values)
