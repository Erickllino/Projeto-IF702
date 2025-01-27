import time 
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from openpyxl import Workbook
from tqdm import tqdm  # Biblioteca para criar barras de progresso

# Define o ponto inicial de treinamento para retomar execuções anteriores (mínimo é 1)
startAt = 1

# Função para salvar os resultados em uma planilha Excel
def save_to_excel(data, filename="CNN_data " + str(time.time())[11:]):
    # Define o caminho e cria o diretório, se necessário
    caminho_pasta = os.path.join("data", filename + ".xlsx")
    directory = os.path.dirname(caminho_pasta)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Cria uma planilha Excel com os resultados
    wb = Workbook()
    ws = wb.active
    fulldata = [["Iteration", "Camadas", "FnAtivação", "FnCusto", "LnRate", "Epocas", "DropOut", "Precision", "Recall", "F1-Score", "Accuracy", "Duração_Execução", "Média_Duração", "Expectativa_Fim"]] + data
    for row in fulldata:
        ws.append([str(x) for x in row])
    wb.save(caminho_pasta)
    print(f"Arquivo salvo com sucesso em: {os.path.abspath(caminho_pasta)}")

# Função para converter rótulos em formato one-hot encoding (necessário para MSELoss)
def one_hot(labels, device, num_classes=10):
    a = torch.eye(num_classes).to(device)  # Cria uma matriz identidade e move para o dispositivo
    return a[labels]  # Retorna apenas as linhas correspondentes aos rótulos

# Função para criar o modelo CNN
def create_cnn(fc_layer_sizes, kernel_size, stride, pooling_kernel, feature_maps, dropout, activation_fn):
    layers = []
    input_channels = 1  # MNIST possui imagens em escala de cinza (1 canal)
    
    # Adiciona camadas convolucionais e operações de pooling
    for fm in feature_maps:
        layers.append(nn.Conv2d(input_channels, fm, kernel_size=kernel_size, stride=stride, padding=1))
        layers.append(activation_fn())  # Função de ativação (ex.: ReLU)
        layers.append(nn.MaxPool2d(kernel_size=pooling_kernel))  # Operação de pooling
        input_channels = fm  # Atualiza os canais de entrada para o próximo mapa de características
    
    layers.append(nn.Flatten())  # Achata a saída da convolução para ser usada em camadas totalmente conectadas

    # Adiciona camadas totalmente conectadas
    input_size = feature_maps[-1] * (28 // (2 ** len(feature_maps))) ** 2
    for size in fc_layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))  # Adiciona dropout se especificado
        input_size = size

    # Adiciona a última camada para classificação em 10 classes (digitos de 0-9)
    layers.append(nn.Linear(input_size, 10))
    return nn.Sequential(*layers)  # Retorna o modelo como um Sequential

# Função para treinar e avaliar o modelo
def train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device):
    model.to(device)  # Move o modelo para GPU ou CPU
    for epoch in tqdm(range(epoch), colour="red"):  # Treinamento por epoch
        model.train()
        for xb, yb in train_loader:  # Loop pelos lotes de treinamento
            xb, yb = xb.to(device), yb.to(device)
            if isinstance(loss_fn, nn.MSELoss):  # Converte rótulos para one-hot se necessário
                yb = one_hot(yb, device)
            optimizer.zero_grad()
            y_pred = model(xb)
            if isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(y_pred, yb.float())
            else:
                loss = loss_fn(y_pred, yb)
           loss.backward()
            optimizer.step()
    
    # Avaliação no conjunto de teste
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb).argmax(dim=1)  # Predição da classe com maior probabilidade
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # Retorna o relatório de classificação (métricas como precisão, recall, F1)
    return classification_report(all_labels, all_preds, output_dict=True)

# Parâmetros configuráveis do modelo
mapaCaracs = [[5]]  # Mapas de características
fullyCons = [[16]]  # Camadas totalmente conectadas
kernels = [2]  # Tamanho do kernel
strides = [1]  # Stride (passo da convolução)
poolings = [2]  # Tamanho do kernel de pooling
# mapaCaracs = [[5], [5, 10], [5, 10, 15]]
# fullyCons = [[16], [16, 32]]
# kernels = [2, 4]
# strides = [1, 2]
# poolings = [2, 4]
dropouts = [0.3]  # Probabilidade de dropout
activation_fns = [nn.ReLU]  # Funções de ativação
learning_rates = [0.0003]  # Taxa de aprendizado
epochs = [5, 15]  # Número de épocas de treinamento
loss_fns = [nn.MSELoss()]  # Funções de perda
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define o dispositivo (GPU/CPU)

# Preparação dos dados do MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loop de treinamento e avaliação para CNN
cnn_reports = []
countCNN = 1  # Contador de execuções
tempoTotal = 0  # Tempo total acumulado
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
                        tempoInicio = time.time()
                        report = train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device)
                        tempoFinal = time.time()
                        tempoExecucao = tempoFinal - tempoInicio
                        tempoTotal += tempoExecucao
                        
                        # Calcula e exibe tempo restante estimado
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
