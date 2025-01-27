import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from openpyxl import Workbook
from tqdm import tqdm #* para criar barras de progresso 

# Configuração inicial: define a iteração inicial (startAt deve ser >= 1)
startAt = 1

# Função para criar e salvar relatórios no Excel
def save_to_excel(data, filename="CNN_data " + str(time.time())[11:]):
    caminho_pasta = os.path.join("data", filename + ".xlsx")
    
    # Verificar se o filename contém um caminho de diretório
    directory = os.path.dirname(caminho_pasta)

    # Se houver diretório especificado e ele não existir, criar
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    wb = Workbook()
    ws = wb.active

    fulldata = [["Iteration", "Camadas", "FnAtivação", "FnCusto", "LnRate", "Epocas", "DropOut", "Precision", "Recall", "F1-Score", "Accuracy", "Duração_Execução", "Média_Duração", "Expectativa_Fim"]] + data

    for row in fulldata:
        ws.append([str(x) for x in row])

    wb.save(caminho_pasta)
    print(f"Arquivo salvo com sucesso em: {os.path.abspath(caminho_pasta)}")

# Função para converter rótulos para one-hot encoding
def one_hot(labels, device, num_classes=10):
    a = torch.eye(num_classes)
    a = a.to(device)
    a = a[labels]
    return a

def create_cnn(fc_layer_sizes, kernel_size, stride, pooling_kernel, feature_maps, dropout, activation_fn):
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
def train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device):
    model.to(device)
    for epoch in tqdm(range(epoch), colour="red"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if isinstance(loss_fn, nn.MSELoss):
                yb = one_hot(yb, device)
            optimizer.zero_grad()
            y_pred = model(xb)
            if isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(y_pred, yb.float())
            else:
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
mapaCaracs = [[5]]
fullyCons = [[16]]
kernels = [2]
strides = [1]
poolings = [2]
# mapaCaracs = [[5], [5, 10], [5, 10, 15]]
# fullyCons = [[16], [16, 32]]
# kernels = [2, 4]
# strides = [1, 2]
# poolings = [2, 4]

dropouts = [0.3]
activation_fns = [nn.ReLU]
learning_rates = [0.0003]
epochs = [5, 15]
loss_fns = [nn.MSELoss()]
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparação dos dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Treinamento e avaliação para CNN
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
                        loss_fn = nn.CrossEntropyLoss()

                        print(f"Treino {countCNN}/{iteracoesTotal} iniciado:")
                        print(f"Filter: {mapaCarac}, FullyCon: {fullyCon}, Kernel: {kernel}, Stride: {stride}, Pooling: {pooling}, Epoch: {epoch}")
                        tempoInicio = time.time()
                        report = train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device)
                        tempoFinal = time.time()
                        tempoExecucao = tempoFinal - tempoInicio
                        print(f"Treino {countCNN}/{iteracoesTotal} finalizado em {tempoExecucao:.2f}s!")
                        
                        # Atualiza estimativas de tempo e salva resultados
                        tempoTotal += tempoExecucao
                        tempoMedio = tempoTotal / countCNN
                        tempoRestante = tempoMedio * (iteracoesTotal - countCNN)
                        horasRestante = tempoRestante // 3600
                        minutosRestante = (tempoRestante % 3600) // 60
                        segundosRestante = tempoRestante % 60
                        print(f"Expectativa de término em: {horasRestante:.0f}h {minutosRestante:.0f}m {segundosRestante:.2f}s.")

                        # Salva resultados no relatório
                        cnn_reports.append([
                            str(countCNN) + "/"+ str(iteracoesTotal),
                            mapaCarac,
                            fullyCon,
                            kernel,
                            stride,
                            pooling,
                            epoch,
                            format(report["weighted avg"]["precision"], ".4f").replace(".", ","),
                            format(report["weighted avg"]["recall"], ".4f").replace(".", ","),
                            format(report["weighted avg"]["f1-score"], ".4f").replace(".", ","),
                            format(report["accuracy"], ".4f").replace(".", ","),
                            format(tempoExecucao, ".4f").replace(".", ","),
                            format(tempoMedio, ".4f").replace(".", ","),
                            format(horasRestante, ".0f") + "h " + format(minutosRestante, ".0f") + "m " + format(segundosRestante, ".2f") + "s",
                        ])
                        if countCNN % 5 == 0:  # Salva relatório parcial a cada 5 iterações
                            save_to_excel(cnn_reports, "CNN_data " + str(countCNN) + " " + str(time.time())[11:] + ".xlsx")
                        
                        countCNN += 1  # Incrementa o contador

# Salva o relatório final
save_to_excel(cnn_reports, filename="CNN_data " + str(countCNN - 1) + " " + str(time.time())[11:])
