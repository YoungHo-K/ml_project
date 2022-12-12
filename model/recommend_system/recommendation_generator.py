import math
import numpy as np

from model.recommend_system.input_params import InputParams
from model.recommend_system.user_item_matrix import UserItemMatrix


class RecommendationGenerator:
    def __init__(self, user_item_matrix=None, number_of_similar_users=None, number_of_recommend_items=None):
        if (user_item_matrix is None) or (number_of_similar_users is None) or (number_of_recommend_items is None):
            raise Exception("[ERROR] Invalid parameters.")

        self.user_item_matrix = user_item_matrix
        self.number_of_similar_users = number_of_similar_users
        self.number_of_recommend_items = number_of_recommend_items

    def generate(self, user_id):
        item_ratings_of_user = self.user_item_matrix.matrix[user_id]

        rated_item_indexes = np.where(item_ratings_of_user != 0)[0]
        mean_item_rating = np.mean(item_ratings_of_user[rated_item_indexes]) if len(rated_item_indexes) != 0 else 0

        similar_user_id_list, user_similarities = self.get_similar_users(user_id)

        recommendation_results = list()
        for item_id, rating in enumerate(item_ratings_of_user):
            if rating != 0:
                continue

            k = 0
            result = 0
            for similar_user, similarity in zip(similar_user_id_list, user_similarities):
                item_rating_of_similar_user = self.user_item_matrix.matrix[similar_user]
                if item_rating_of_similar_user[item_id] == 0:
                    continue

                rated_item_indexes_of_similar_user = np.where(item_rating_of_similar_user != 0)[0]
                mean_item_rating_of_similar_user = np.mean(item_rating_of_similar_user[rated_item_indexes_of_similar_user]) if len(rated_item_indexes_of_similar_user) != 0 else 0

                k += similarity
                result += similarity * (item_rating_of_similar_user[item_id] - mean_item_rating_of_similar_user)

            if k != 0:
                predicted_rating = mean_item_rating + (1 / k) * result
                recommendation_results.append((item_id, predicted_rating))

        recommendation_results.sort(key=lambda x: x[1])
        recommendation_results = recommendation_results[::-1][: self.number_of_recommend_items]
        output = list()
        for item_id, _ in recommendation_results:
            # output.append(f"{item_id + 1}")
            output.append(item_id + 1)

        # output = " ".join(output)

        return output

    def get_similar_users(self, user_id):
        user_similarities = self.user_item_matrix.user_similarity_matrix[user_id]

        similar_user_id_list = np.argsort(user_similarities)[::-1]
        similar_user_id_list = np.delete(similar_user_id_list, np.where(similar_user_id_list == user_id)[0])
        similar_user_id_list = similar_user_id_list[: self.number_of_similar_users]

        return similar_user_id_list, user_similarities[similar_user_id_list]


def ndcg(gt, rec):
    idcg = sum([1.0 / math.log(i + 2, 2) for i in range(len(gt))])
    dcg = 0.0
    for i, r in enumerate(rec):
        if r not in gt:
            continue
        gt_index = gt.index(r)
        if i != gt_index:
            rel = 0.7
        else:
            rel = 1.0
        dcg += rel / math.log(i + 2, 2)
    ndcg = dcg / idcg

    return ndcg


def get_outputs(file_path):
    outputs = list()
    with open(file_path, "r") as file_descriptor:
        lines = file_descriptor.readlines()

        for values in lines:
            values = list(map(int, values.split(" ")))

            outputs.append(values)

    return outputs


if __name__ == "__main__":
    import os

    ROOT_DIR_PATH = "/Users/youngho/Documents/GitHub/ml_project"
    INPUT_FILE_PATH = "model/recommend_system/testcase/input/input006.txt"
    OUTPUT_FILE_PATH = "model/recommend_system/testcase/output/output006.txt"

    outputs = get_outputs(os.path.join(ROOT_DIR_PATH, OUTPUT_FILE_PATH))

    input_params = InputParams.get_params(file_path=os.path.join(ROOT_DIR_PATH, INPUT_FILE_PATH))

    user_item_matrix = UserItemMatrix()
    user_item_matrix.generate(input_params)

    rec_sys = RecommendationGenerator(user_item_matrix, input_params.number_of_similar_users, input_params.number_of_recommend_items)
    for user_id, output in zip(input_params.recommend_user_list, outputs):
        recommendation_results = rec_sys.generate(user_id - 1)

        score = ndcg(output, recommendation_results)

        print(score)

