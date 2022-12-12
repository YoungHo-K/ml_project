import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.efficientnet import EfficientNet_B0_Weights


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()

        self.backbone = models.efficientnet_b0(EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.embedding = nn.Linear(in_features=1000, out_features=512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)

        return x


class TabularFeatureExtractor(nn.Module):
    def __init__(self):
        super(TabularFeatureExtractor, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_features=23, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512)
        )

    def forward(self, x):
        x = self.embedding(x)

        return x


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()

        self.image_feature_extractor = ImageFeatureExtractor()
        self.tabular_feature_extractor = TabularFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, image, tabular):
        image_feature = self.image_feature_extractor(image)
        tabular_feature = self.tabular_feature_extractor(tabular)
        feature = torch.cat([image_feature, tabular_feature], dim=-1)

        output = self.classifier(feature)

        return output