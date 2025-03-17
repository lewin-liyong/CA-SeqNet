# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Definition of CBAM module based on 1D feature map
class ComplementaryAttentionNetwork1D(nn.Module):
    def __init__(self, in_channels, inner_units_ratio=1):
        super(ComplementaryAttentionNetwork1D, self).__init__()

        self.inner_units_ratio = inner_units_ratio
        self.in_channels = in_channels

        # Channel attention: two layers of 1x1 1D convolution
        self.fc1 = nn.Conv1d(in_channels, in_channels * inner_units_ratio, kernel_size=1, stride=1)
        self.fc2 = nn.Conv1d(in_channels * inner_units_ratio, in_channels, kernel_size=1, stride=1)

    def forward(self, feature_map):
        feature_map_shape = feature_map.size()

        # Channel attention part
        globel_avg = F.adaptive_avg_pool1d(feature_map, 1)  # Global average pooling
        globel_max = F.adaptive_max_pool1d(feature_map, 1)  # Global maximum pooling

        # Expand the results of global average pooling and maximum pooling
        channel_avg_weights = globel_avg.view(feature_map_shape[0], self.in_channels, 1)
        channel_max_weights = globel_max.view(feature_map_shape[0], self.in_channels, 1)

        # Concatenate the results of average and maximum pooling
        channel_w_reshape = torch.cat([channel_avg_weights, channel_max_weights], dim=2)

        # Generate channel attention through two layers of fully connected convolution
        fc_1 = F.relu(self.fc1(channel_w_reshape))
        fc_2 = torch.sigmoid(self.fc2(fc_1))

        # Calculate channel attention and apply it to feature maps
        channel_attention = torch.sum(fc_2, dim=2, keepdim=True)
        channel_attention = torch.sigmoid(channel_attention)
        feature_map_with_channel_attention = feature_map * channel_attention

        return feature_map_with_channel_attention


# Backbone network structure
class VGGWithCBAM1D_CA(nn.Module):
    def __init__(self, num_classes=3):
        super(VGGWithCBAM1D_CA, self).__init__()

            # Block 1,2
        self.vgg = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 6
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 7
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # CBAM module based on 1D
        self.cbam = ComplementaryAttentionNetwork1D(in_channels=256)

        # Defining the classifier part
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256 * 78, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, num_classes)
        )

    def forward(self, x):
        x = self.vgg(x)
        print(x.shape)
        # Added 1D-based CBAM module
        x = self.cbam(x)

        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGGWithCBAM1D_CA()
    a = torch.Tensor(12, 1, 10000)  # The input dimensions are (n_components,batch_size,sequence_length)
    b = model(a)  #output classification result
