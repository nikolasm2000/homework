import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A CNN model for image classification.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(4704, 120)
        self.fc3 = nn.Linear(120, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dataset.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
