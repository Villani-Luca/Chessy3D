import torch
from posthog.exception_utils import epoch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Runner:
    @staticmethod
    def train_ae(
            model: AE,
            epochs = 20,
            loss = nn.MSELoss()):

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        outputs = []
        losses = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(epochs):
            for images, _ in loader:
                images = images.view(-1, 28 * 28).to(device)

                reconstructed = model(images)
                loss = loss_function(reconstructed, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            outputs.append((epoch, images, reconstructed))
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")