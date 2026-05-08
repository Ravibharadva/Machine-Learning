import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)


class LogisticNN(nn.Module):
    def __init__(self, n_features):
        super(LogisticNN, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train(X_train, y_train, epochs=200, lr=0.01, batch_size=64):
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model     = LogisticNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg = epoch_loss / len(X_train)
        losses.append(avg)
        if epoch % 50 == 0:
            print(f"  Epoch [{epoch:>3}/200]  BCE Loss: {avg:.4f}")

    return model, losses


def predict(model, X_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    probs = np.atleast_1d(probs)
    return (probs >= threshold).astype(int), probs