import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import keras

# Dataset personalizado para PyTorch
class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Classes da CNN
class CNN_Model(nn.Module):
    def __init__(self, kernel_size, stride, pool_size, num_features, num_classes=10):
        super(CNN_Model, self).__init__()

        self.conv1 = nn.Conv2d(1, num_features, kernel_size=kernel_size, stride=stride, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_features * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN_2(CNN_Model):
    def __init__(self, kernel_size, stride, pool_size, num_features, num_classes=10):
        super(CNN_2, self).__init__(kernel_size, stride, pool_size, num_features, num_classes)
        self.conv2 = nn.Conv2d(num_features, num_features * 2, kernel_size=kernel_size, stride=stride, padding=1)
        self.fc = nn.Linear((num_features * 2) * 3 * 3, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Função de treinamento
def train_model(model, train_dl, optimizer, loss_fn, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Avaliação do modelo
def evaluate_model(model, test_dl):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(yb.tolist())
            y_pred.extend(preds.tolist())
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

# Parâmetros para grid search
kernel_sizes = [2, 4, 8]
strides = [1, 2]
pool_sizes = [2, 4]
num_features = [5, 10, 15]
num_layers = [1, 2]
learning_rates = [3e-5, 3e-2, 3e-10]
num_epochs = [20, 50, 80, 100]
loss_functions = [nn.CrossEntropyLoss(), nn.MSELoss()]

# Carregando MNIST
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
train_ds = MNISTDataset(train_X, train_y)
test_ds = MNISTDataset(test_X, test_y)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=256)

# Grid Search
results = []
count = 1

for kernel_size in kernel_sizes:
    for stride in strides:
        for pool_size in pool_sizes:
            for features in num_features:
                for num_layer in num_layers:
                    for lr in learning_rates:
                        for loss_fn in loss_functions:
                            for ep in num_epochs:
                                if num_layer == 1:
                                    model = CNN_Model(kernel_size, stride, pool_size, features)
                                else:
                                    model = CNN_2(kernel_size, stride, pool_size, features)

                                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                                train_model(model, train_dl, optimizer, loss_fn, ep)

                                report = evaluate_model(model, test_dl)

                                results.append({
                                    "Iteration": count,
                                    "Kernel Size": kernel_size,
                                    "Stride": stride,
                                    "Pool Size": pool_size,
                                    "Features": features,
                                    "Layers": num_layer,
                                    "Learning Rate": lr,
                                    "Loss Function": loss_fn.__class__.__name__,
                                    "Epochs": ep,
                                    "Precision": report["weighted avg"]["precision"],
                                    "Recall": report["weighted avg"]["recall"],
                                    "F1-Score": report["weighted avg"]["f1-score"],
                                    "Accuracy": report["accuracy"],
                                })

                                print(f"Iteration {count} completed")
                                count += 1

# Salvando os resultados em Excel
df = pd.DataFrame(results)
df.to_excel("cnn_grid_search_results.xlsx", index=False, engine="openpyxl")

print("Grid search concluído e resultados salvos em cnn_grid_search_results.xlsx.")
