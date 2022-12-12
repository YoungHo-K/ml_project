import os
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A

from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from dacon.dataset.dataset import MultiModalDataset
from dacon.neural_network.classifier import Classifier
from dacon.neural_network.architecture import ClassificationModel

warnings.filterwarnings(action="ignore")


def set_seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_values(x):
    return x.values.reshape(-1, 1)


if __name__ == "__main__":
    ROOT_DIR_PATH = "/Users/youngho/Documents/GitHub/ml_project/dacon/resource"
    TRAIN_FILE_NAME = "train.csv"
    TEST_FILE_NAME = "test.csv"

    EPOCH = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUMBER_OF_TILES = 100
    SIZE_OF_TILE = 128
    NUMERIC_COLUMNS = ["나이", "암의 장경", "ER_Allred_score", "PR_Allred_score", "KI-67_LI_percent", "HER2_SISH_ratio"]
    IGNORE_COLUMNS = ["ID", "img_path", "mask_path", "수술연월일", "N_category"]

    set_seed_everything(seed=4570)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data_frame = pd.read_csv(os.path.join(ROOT_DIR_PATH, TRAIN_FILE_NAME))
    test_data_frame = pd.read_csv(os.path.join(ROOT_DIR_PATH, TEST_FILE_NAME))

    for index in range(0, len(train_data_frame)):
        train_data_frame["img_path"].iloc[index] = os.path.join(ROOT_DIR_PATH, train_data_frame["img_path"].iloc[index][2:])

    for index in range(0, len(test_data_frame)):
        test_data_frame["img_path"].iloc[index] = os.path.join(ROOT_DIR_PATH, test_data_frame["img_path"].iloc[index][2:])

    train_data_frame["암의 장경"] = train_data_frame["암의 장경"].fillna(train_data_frame["암의 장경"].mean())
    train_data_frame = train_data_frame.fillna(0)
    test_data_frame["암의 장경"] = test_data_frame["암의 장경"].fillna(train_data_frame["암의 장경"].mean())
    test_data_frame = test_data_frame.fillna(0)

    for column in train_data_frame.columns:
        if column in IGNORE_COLUMNS:
            continue

        if column in NUMERIC_COLUMNS:
            scaler = StandardScaler()

            train_data_frame[column] = scaler.fit_transform(get_values(train_data_frame[column]))
            test_data_frame[column] = scaler.transform(get_values(test_data_frame[column]))

        else:
            le = LabelEncoder()

            train_data_frame[column] = le.fit_transform(get_values(train_data_frame[column]))
            test_data_frame[column] = le.transform(get_values(test_data_frame[column]))

    train_labels = train_data_frame["N_category"]
    train_data_frame = train_data_frame.drop(columns=["N_category"])

    train_transform = A.Compose([A.Transpose(p=0.5),
                                 A.VerticalFlip(p=0.5),
                                 A.HorizontalFlip(p=0.5),
                                 ToTensorV2()])
    test_transform = A.Compose([ToTensorV2()])

    classifier_ensembles = list()
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for index, (train_index, test_index) in enumerate(kfold.split(train_data_frame, train_labels)):
        X_train_data = train_data_frame.iloc[train_index, :]
        y_train_data = train_labels.iloc[train_index]
        X_valid_data = train_data_frame.iloc[test_index, :]
        y_valid_data = train_labels.iloc[test_index]

        train_dataset = MultiModalDataset(medical_data_frame=X_train_data,
                                          labels=y_train_data.values,
                                          transform=train_transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        validation_dataset = MultiModalDataset(medical_data_frame=X_valid_data,
                                               labels=y_valid_data.values,
                                               transform=test_transform)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = nn.DataParallel(ClassificationModel())
        model.eval()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1,
                                                               threshold_mode="abs", min_lr=1e-8, verbose=True)
        criterion = nn.BCEWithLogitsLoss()

        classifier = Classifier(model=model, optimizer=optimizer, scheduler=scheduler,
                                criterion=criterion, threshold=0.5, device=device)

        classifier.train(train_data_loader=train_loader, validation_data_loader=validation_loader, epoch=EPOCH)

        classifier_ensembles.append(classifier)

    test_dataset = MultiModalDataset(medical_data_frame=test_data_frame,
                                     labels=None,
                                     transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ensemble_y_pred_list = list()
    for classifier in classifier_ensembles:
        y_pred_list = classifier.inference(test_loader)

        ensemble_y_pred_list.append(y_pred_list)

    ensemble_y_pred_list = np.sum(ensemble_y_pred_list, axis=0)
    ensemble_y_pred_list = [1 if x >= 3 else 0 for x in ensemble_y_pred_list]

    submit = pd.read_csv(os.path.join(ROOT_DIR_PATH, "sample_submission.csv"))
    submit['N_category'] = ensemble_y_pred_list
    submit.to_csv(os.path.join(ROOT_DIR_PATH, "submit.csv"), index=False)