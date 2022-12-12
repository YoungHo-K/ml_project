import numpy as np


class InputParams:
    def __init__(self):
        self.number_of_total_users = None
        self.number_of_total_items = None
        self.number_of_similar_users = None
        self.number_of_recommend_items = None
        self.recommend_user_list = list()
        self.user_item_rating_list = list()

    @staticmethod
    def get_params(file_path=None):
        if file_path is None:
            raise Exception(f"[ERROR] Invalid file path. {file_path}")

        with open(file_path, "r") as file_descriptor:
            lines = file_descriptor.readlines()

        input_params = InputParams()
        input_params.number_of_similar_users = int(lines[0])
        input_params.number_of_recommend_items = int(lines[1])
        input_params.number_of_total_users = int(lines[2])
        input_params.number_of_total_items = int(lines[3])
        for index in range(0, int(lines[4])):
            user, item, rating = lines[5 + index].split(" ")

            input_params.user_item_rating_list.append((int(user), int(item), float(rating)))

        recommend_users_index = 5 + len(input_params.user_item_rating_list)
        for index in range(recommend_users_index + 1, len(lines)):
            input_params.recommend_user_list.append(int(lines[index]))

        return input_params

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = "-------------- Input Params --------------"
        msg += f"\n Number of total users:               {self.number_of_total_users}"
        msg += f"\n Number of total items:               {self.number_of_total_items}"
        msg += f"\n Number of similar users:             {self.number_of_similar_users}"
        msg += f"\n Number of recommend items:           {self.number_of_recommend_items}"
        msg += f"\n Recommend user list:                 {len(self.recommend_user_list)}"
        msg += f"\n Length of user-item rating data:     {len(self.user_item_rating_list)}"
        msg += "\n\n"

        return msg
