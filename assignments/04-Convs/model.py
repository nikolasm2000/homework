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
        self.conv1 = nn.Conv2d(num_channels, 6, 5, stride = 2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_classes)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dataset.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
