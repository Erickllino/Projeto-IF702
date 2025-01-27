import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from openpyxl import Workbook
from tqdm import tqdm

# Define o ponto inicial de treinamento para retomar execuções anteriores (mínimo é 1)
startAt = 1

def save_to_excel(data, filename="CNN_data" + str(time.time())[11:]):
    from pathlib import Path
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / f"{filename}.xlsx"
    wb = Workbook()
    ws = wb.active
    fulldata = [["Iteration", "Camadas", "FnAtivação", "FnCusto", "LnRate", "Epocas", "DropOut", "Precision", "Recall", "F1-Score", "Accuracy", "Duração_Execução", "Média_Duração", "Expectativa_Fim"]] + data
    for row in fulldata:
        ws.append([str(x) for x in row])
    wb.save(filepath)
    print(f"Arquivo salvo com sucesso em: {filepath.resolve()}")

# Função para criar o modelo CNN
def create_cnn(fc_layer_sizes, kernel_size, stride, pooling_kernel, feature_maps, dropout, activation_fn):
    layers = []
    input_channels = 1
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

# Função para treinar e avaliar o modelo
def train_and_evaluate(model, train_loader, test_loader, epochs, loss_fn, optimizer, device):
    scaler = torch.amp.GradScaler("cuda")
    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs", colour="red"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                y_pred = model(xb)
                loss = loss_fn(y_pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast("cuda"):
                y_pred = model(xb).argmax(dim=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return classification_report(all_labels, all_preds, output_dict=True)

# Configurações do modelo
mapaCaracs = [[5], [5, 10]]
fullyCons = [[16], [16, 32]]
kernels = [2]
strides = [1]
poolings = [2]
dropouts = [0.3]
activation_fns = [nn.ReLU]
learning_rates = [0.0003]
epochs = [5, 15]
loss_fns = [nn.CrossEntropyLoss()]
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparação dos dados do MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Loop de treinamento e avaliação para CNN
cnn_reports = []
countCNN = 1
tempoTotal = 0
iteracoesTotal = len(mapaCaracs) * len(fullyCons) * len(kernels) * len(strides) * len(poolings) * len(epochs)

for mapaCarac in mapaCaracs:
    for fullyCon in fullyCons:
        for kernel in kernels:
            for stride in strides:
                for pooling in poolings:
                    for epoch in epochs:
                        if countCNN < startAt:
                            countCNN += 1
                            continue

                        model = create_cnn(fullyCon, kernel, stride, pooling, mapaCarac, dropouts[0], activation_fns[0])
                        optimizer = optim.Adam(model.parameters(), lr=learning_rates[0])
                        loss_fn = loss_fns[0]

                        print(f"Treino {countCNN}/{iteracoesTotal} iniciado:")
                        tempoInicio = time.time()
                        report = train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device)
                        tempoExecucao = time.time() - tempoInicio
                        tempoTotal += tempoExecucao

                        tempoMedio = tempoTotal / countCNN
                        tempoRestante = tempoMedio * (iteracoesTotal - countCNN)
                        horasRestante = tempoRestante // 3600
                        minutosRestante = (tempoRestante % 3600) // 60
                        segundosRestante = tempoRestante % 60

                        cnn_reports.append([
                            f"{countCNN}/{iteracoesTotal}", mapaCarac, fullyCon, kernel, stride, pooling, epoch,
                            f"{report['weighted avg']['precision']:.4f}",
                            f"{report['weighted avg']['recall']:.4f}",
                            f"{report['weighted avg']['f1-score']:.4f}",
                            f"{report['accuracy']:.4f}",
                            f"{tempoExecucao:.4f}",
                            f"{tempoMedio:.4f}",
                            f"{horasRestante:.0f}h {minutosRestante:.0f}m {segundosRestante:.2f}s",
                        ])

                        if countCNN % 5 == 0:
                            save_to_excel(cnn_reports, f"CNN_data_{countCNN}")

                        countCNN += 1

save_to_excel(cnn_reports, filename=f"CNN_data_final_{countCNN - 1}")
