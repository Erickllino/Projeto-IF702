import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from openpyxl import Workbook

# Função para criar e salvar relatórios no Excel
def save_to_excel(data, filename="classification_report.xlsx"):
    wb = Workbook()
    ws = wb.active

    for row in data:
        ws.append(row)

    wb.save(filename)

# Definição das arquiteturas para MLP e CNN
def create_mlp(layer_sizes, dropout, activation_fn):
    layers = []
    input_size = 28 * 28  # Tamanho das imagens do MNIST
    for size in layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_size = size
    layers.append(nn.Linear(input_size, 10))  # Camada de saída para 10 classes
    return nn.Sequential(*layers)

def create_cnn(conv_layers, fc_layer_sizes, kernel_size, stride, pooling_kernel, feature_maps, dropout, activation_fn):
    layers = []
    input_channels = 1  # MNIST tem 1 canal (grayscale)

    for fm in feature_maps:
        layers.append(nn.Conv2d(input_channels, fm, kernel_size=kernel_size, stride=stride, padding=1))
        layers.append(activation_fn())
        layers.append(nn.MaxPool2d(kernel_size=pooling_kernel))
        input_channels = fm

    layers.append(nn.Flatten())
    input_size = feature_maps[-1] * (28 // (2 ** len(feature_maps))) ** 2

    for size in fc_layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_size = size

    layers.append(nn.Linear(input_size, 10))
    return nn.Sequential(*layers)

# Função para treinar e testar o modelo
def train_and_evaluate(model, train_loader, test_loader, epochs, loss_fn, optimizer, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()

    # Teste
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb).argmax(dim=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return classification_report(all_labels, all_preds, output_dict=True)

# Parâmetros configuráveis
layer_configs = [[128, 64], [256, 128, 64]]  # Configurações para camadas
conv_configs = [
    ([16, 32], [128], 3, 1, 2),  # (Camadas de convolução, FC, kernel, stride, pooling)
    ([32, 64], [256, 128], 3, 1, 2),
]
dropout = 0.2
activation_fn = nn.ReLU
learning_rate = 0.001
epochs = 5
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparação dos dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Treinamento e avaliação para MLP
mlp_reports = []
for layer_sizes in layer_configs:
    model = create_mlp(layer_sizes, dropout, activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    report = train_and_evaluate(model, train_loader, test_loader, epochs, loss_fn, optimizer, device)
    mlp_reports.append([f"MLP-{layer_sizes}", report['accuracy']])

# Treinamento e avaliação para CNN
cnn_reports = []
for conv_layers, fc_layers, kernel_size, stride, pooling_kernel in conv_configs:
    feature_maps = conv_layers
    model = create_cnn(fc_layers, kernel_size, stride, pooling_kernel, feature_maps, dropout, activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    report = train_and_evaluate(model, train_loader, test_loader, epochs, loss_fn, optimizer, device)
    cnn_reports.append([f"CNN-{conv_layers}-{fc_layers}", report['accuracy']])

# Salvar relatórios em planilha Excel
combined_reports = [["Model", "Accuracy"]] + mlp_reports + cnn_reports
save_to_excel(combined_reports)
