# Importação de bibliotecas necessárias
import time  # Trabalhar com medição de tempo
import torch  # Framework para machine learning
import torch.nn as nn  # Módulo para redes neurais
import torch.optim as optim  # Otimizadores
import os  # Operações com o sistema de arquivos
from torch.utils.data import DataLoader  # Gerenciamento de dados
from torchvision import datasets, transforms  # Trabalhar com datasets e transformações
from sklearn.metrics import classification_report  # Métricas de avaliação
from openpyxl import Workbook  # Manipulação de arquivos Excel
from tqdm import tqdm  # Barra de progresso

# Configuração inicial: define a iteração inicial (startAt deve ser >= 1)
startAt = 481

# Função para salvar dados em um arquivo Excel
def save_to_excel(data, filename="MLP_data " + str(time.time())[11:]):
    # Define o caminho para salvar o arquivo
    caminho_pasta = os.path.join("data", filename + ".xlsx")
    
    # Cria o diretório se ele não existir
    directory = os.path.dirname(caminho_pasta)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Cria uma planilha e adiciona os dados
    wb = Workbook()
    ws = wb.active
    # Cabeçalhos dos dados
    fulldata = [["Iteration", "Camadas", "FnAtivação", "FnCusto", "LnRate", "Epocas", "DropOut", "Precision", "Recall", "F1-Score", "Accuracy", "Duração_Execução", "Média_Duração", "Expectativa_Fim"]] + data
    # Adiciona os dados linha por linha
    for row in fulldata:
        ws.append([str(x) for x in row])
    # Salva o arquivo
    wb.save(caminho_pasta)
    print(f"Arquivo salvo com sucesso em: {os.path.abspath(caminho_pasta)}")

# Função para realizar one-hot encoding nos rótulos
def one_hot(labels, device, num_classes=10):
    a = torch.eye(num_classes)  # Cria uma matriz identidade (número de classes)
    a = a.to(device)  # Move para o dispositivo (CPU ou GPU)
    a = a[labels]  # Seleciona os rótulos correspondentes
    return a

# Função para criar uma MLP com base em parâmetros
def create_mlp(layer_sizes, dropout, activation_fn):
    layers = []
    layers.append(nn.Flatten())  # Achata a entrada (imagens 28x28 -> vetor de 784)
    input_size = 28 * 28  # Tamanho das imagens MNIST
    for size in layer_sizes:
        layers.append(nn.Linear(input_size, size))  # Adiciona camada densa
        layers.append(activation_fn())  # Adiciona função de ativação
        if dropout > 0:
            layers.append(nn.Dropout(dropout))  # Adiciona dropout (se especificado)
        input_size = size
    layers.append(nn.Linear(input_size, 10))  # Camada final (10 classes)
    return nn.Sequential(*layers)  # Retorna a rede sequencial

# Função para treinar e avaliar o modelo
def train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device):
    model.to(device)  # Move o modelo para o dispositivo (CPU ou GPU)
    for epoch in tqdm(range(epoch), colour="red"):  # Loop de treinamento para cada época
        model.train()  # Coloca o modelo em modo de treinamento
        for xb, yb in train_loader:  # Itera sobre os dados de treinamento
            xb, yb = xb.to(device), yb.to(device)  # Move dados para o dispositivo
            if isinstance(loss_fn, nn.MSELoss):  # Converte rótulos se for MSE
                yb = one_hot(yb, device)
            optimizer.zero_grad()  # Zera os gradientes
            y_pred = model(xb)  # Faz previsão
            if isinstance(loss_fn, nn.MSELoss):  # Calcula perda
                loss = loss_fn(y_pred, yb.float())
            else:
                loss = loss_fn(y_pred, yb)
            loss.backward()  # Calcula gradientes
            optimizer.step()  # Atualiza os pesos

    # Avaliação do modelo
    model.eval()  # Coloca o modelo em modo de avaliação
    all_preds, all_labels = [], []  # Listas para armazenar previsões e rótulos reais
    with torch.no_grad():  # Desativa gradientes
        for xb, yb in test_loader:  # Itera sobre os dados de teste
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb).argmax(dim=1)  # Obtém as classes previstas
            all_preds.extend(y_pred.cpu().numpy())  # Salva previsões
            all_labels.extend(yb.cpu().numpy())  # Salva rótulos reais

    return classification_report(all_labels, all_preds, output_dict=True)  # Retorna relatório

# Parâmetros configuráveis
layer_configs = [[20], [20, 40], [20, 40, 80]]  # Configuração de camadas
dropouts = [0, 0.2, 0.5]  # Taxas de dropout
activation_fns = [nn.ReLU, nn.Sigmoid, nn.Tanh]  # Funções de ativação
learning_rates = [0.0003, 0.00003, 0.000003]  # Taxas de aprendizado
epochs = [3, 7, 10]  # Quantidade de épocas
loss_fns = [nn.MSELoss(), nn.CrossEntropyLoss()]  # Funções de perda
batch_size = 64  # Tamanho do batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Seleção de dispositivo

# Prepara os datasets MNIST com normalização e carrega os dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inicializa variáveis para relatórios e controle
mlp_reports = []
countMLP = 1  # Contador para iterações
tempoTotal = 0  # Tempo total de execução
iteracoesTotal = len(layer_configs) * len(dropouts) * len(activation_fns) * len(learning_rates) * len(loss_fns) * len(epochs)  # Total de iterações

# Loop principal para treinar e avaliar diferentes configurações
for layer_sizes in layer_configs:
    for dropout in dropouts:
        for activation_fn in activation_fns:
            for learning_rate in learning_rates:
                for epoch in epochs:
                    if countMLP < startAt:  # Pula até alcançar a iteração inicial
                        countMLP += len(loss_fns)
                        continue
                    
                    model = create_mlp(layer_sizes, dropout, activation_fn)  # Cria o modelo
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Define otimizador

                    for loss_fn in loss_fns:
                        if countMLP < startAt:  # Pula até alcançar a iteração inicial
                            countMLP += 1
                            continue
                        
                        # Início do treinamento e avaliação
                        tempoInicio = time.time()
                        print(f"Treino {countMLP}/{iteracoesTotal} iniciado:")
                        print(f"Layers: {layer_sizes}, FnAtiv: {activation_fn.__name__}, Dropout: {dropout:.4f}, LnRate: {learning_rate:.4f}, Epoch: {epoch}, LossFn: {loss_fn.__class__.__name__}")
                        report = train_and_evaluate(model, train_loader, test_loader, epoch, loss_fn, optimizer, device)
                        tempoFinal = time.time()
                        tempoExecucao = tempoFinal - tempoInicio  # Calcula tempo de execução
                        print(f"Treino {countMLP}/{iteracoesTotal} finalizado em {tempoExecucao:.2f}s!")

                        # Atualiza estimativas de tempo e salva resultados
                        tempoTotal += tempoExecucao
                        tempoMedio = tempoTotal / countMLP
                        tempoRestante = tempoMedio * (iteracoesTotal - countMLP)
                        horasRestante = tempoRestante // 3600
                        minutosRestante = (tempoRestante % 3600) // 60
                        segundosRestante = tempoRestante % 60
                        print(f"Expectativa de término em: {horasRestante:.0f}h {minutosRestante:.0f}m {segundosRestante:.2f}s.")

                        # Salva resultados no relatório
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
                        if countMLP % 5 == 0:  # Salva relatório parcial a cada 5 iterações
                            save_to_excel(mlp_reports, "MLP_data " + str(countMLP) + " " + str(time.time())[11:] + ".xlsx")
                        
                        countMLP += 1  # Incrementa o contador

# Salva o relatório final
save_to_excel(mlp_reports, filename="MLP_data " + str(countMLP - 1) + " " + str(time.time())[11:])
