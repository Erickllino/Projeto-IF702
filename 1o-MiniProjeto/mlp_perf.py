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

startAt = 485 # startAt >= 1

# Função para criar e salvar relatórios no Excel
def save_to_excel(data, filename="MLP_data " + str(time.time())[11:]):
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

# Definição das arquiteturas para MLP e CNN
def create_mlp(layer_sizes, dropout, activation_fn):
    layers = []
    layers.append(nn.Flatten())  # Garante que a entrada seja achatada
    input_size = 28 * 28  # Tamanho das imagens do MNIST
    for size in layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_size = size
    layers.append(nn.Linear(input_size, 10))  # Camada de saída para 10 classes
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
layer_configs = [[20], [20, 40], [20, 40, 80]]  # Configurações para camadas
dropouts = [0, 0.2, 0.5]
activation_fns = [nn.ReLU, nn.Sigmoid, nn.Tanh]
learning_rates = [0.0003, 0.00003, 0.000003]
epochs = [3, 7, 10]
loss_fns = [nn.MSELoss(), nn.CrossEntropyLoss()]
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
countMLP = 1
tempoTotal = 0
iteracoesTotal = len(layer_configs)*len(dropouts)*len(activation_fns)*len(learning_rates)*len(loss_fns)*len(epochs)

for layer_sizes in layer_configs:
    for dropout in dropouts:
        for activation_fn in activation_fns:
            for learning_rate in learning_rates:
                for epoch in epochs:
                    if countMLP < startAt:
                        countMLP = countMLP + len(loss_fns)
                        continue
                    
                    model = create_mlp(layer_sizes, dropout, activation_fn)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    for loss_fn in loss_fns:
                        if countMLP < startAt:
                            countMLP = countMLP + 1
                            continue
                        
                        tempoInicio = time.time()
                        print(f"Treino {countMLP}/{iteracoesTotal} iniciado:")
                        print(f"Layers: {layer_sizes}, FnAtiv: {activation_fn.__name__}, Dropout: {dropout:.4f}, LnRate: {learning_rate:.4f}, epoch: {epoch}, LossFn: {loss_fn.__class__.__name__}")
                        report = train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device)
                        tempoFinal = time.time()
                        tempoExecucao = tempoFinal - tempoInicio
                        print(f"Treino {countMLP}/{iteracoesTotal} finalizado em {tempoExecucao:.2f}s!")
                        tempoTotal = tempoTotal + tempoExecucao
                        tempoMedio = tempoTotal / countMLP
                        tempoRestante = tempoMedio * (iteracoesTotal - countMLP)
                        horasRestante = tempoRestante // 3600
                        minutosRestante = (tempoRestante % 3600) // 60
                        segundosRestante = tempoRestante % 60
                        print(f"Expectativa de término em: {horasRestante:.0f}h {minutosRestante:.0f}m {segundosRestante:.2f}s.")

                        mlp_reports.append([
                            str(countMLP) + "/"+ str(iteracoesTotal),
                            layer_sizes,
                            activation_fn.__name__,
                            loss_fn.__class__.__name__,
                            learning_rate,
                            epoch,
                            dropout,
                            format(report["weighted avg"]["precision"], ".4f").replace(".", ","),
                            format(report["weighted avg"]["recall"], ".4f").replace(".", ","),
                            format(report["weighted avg"]["f1-score"], ".4f").replace(".", ","),
                            format(report["accuracy"], ".4f").replace(".", ","),
                            format(tempoExecucao, ".4f").replace(".", ","),
                            format(tempoMedio, ".4f").replace(".", ","),
                            format(horasRestante, ".0f") + "h " + format(minutosRestante, ".0f") + "m " + format(segundosRestante, ".2f") + "s",
                        ])
                        if countMLP % 5 == 0:
                            save_to_excel(mlp_reports, "MLP_data " + str(countMLP) + " " + str(time.time())[11:] + ".xlsx")
                            
                        countMLP = countMLP + 1

# Salvar relatórios em planilha Excel
save_to_excel(mlp_reports, filename="MLP_data " + str(countMLP - 1) + " " + str(time.time())[11:] + ".xlsx")
