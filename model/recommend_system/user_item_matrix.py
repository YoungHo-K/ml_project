import numpy as np


class UserItemMatrix:
    def __init__(self):
        self.matrix = None
        self.user_similarity_matrix = None
        self.mean_item_ratings_by_user = None

    def generate(self, params):
        self.set_user_item_matrix(params)
        self.set_user_similiarity()

    def set_user_item_matrix(self, params):
        user_item_matrix = np.zeros((params.number_of_total_users, params.number_of_total_items), dtype=np.float)
        for user_id, item_id, rating in params.user_item_rating_list:
            user_item_matrix[user_id - 1, item_id - 1] = rating

        self.matrix = user_item_matrix

    def set_user_similiarity(self):
        user_similiarity_matrix = np.zeros((self.matrix.shape[0], self.matrix.shape[0]), dtype=np.float)

        for user in range(0, self.matrix.shape[0]):
            item_ratings_by_user = self.matrix[user]
            item_ratings_indexes = np.where(item_ratings_by_user != 0)[0]
            if len(item_ratings_indexes) == 0:
                continue

            for next_user in range(user, self.matrix.shape[0]):
                item_ratings_by_next_user = self.matrix[next_user]
                next_item_ratings_indexes = np.where(item_ratings_by_next_user != 0)[0]
                if len(next_item_ratings_indexes) == 0:
                    continue

                common_item_ratings = np.intersect1d(item_ratings_indexes, next_item_ratings_indexes)
                if len(common_item_ratings) == 0:
                    continue

                similarity = np.dot(item_ratings_by_user[common_item_ratings], item_ratings_by_next_user[common_item_ratings]) / \
                             (np.linalg.norm(item_ratings_by_user[item_ratings_indexes]) * np.linalg.norm(item_ratings_by_next_user[next_item_ratings_indexes]))

                user_similiarity_matrix[user, next_user] = similarity
                user_similiarity_matrix[next_user, user] = similarity

        self.user_similarity_matrix = user_similiarity_matrix
