
#$ Bibliotecas Básicas
import time
import matplotlib.pyplot as plt #* para plotar gráficos
from tqdm import tqdm #* para criar barras de progresso 
from sklearn.metrics import classification_report #* para obter dados estatíticos de eficiência dos resultados
import keras #* para obter o banco de dados necessário para o treinamento
import pandas as pd
#! import seaborn as sns

#$ FrameWork para rede neural utilizado -> Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #* para manipular o banco de dados

#$ Classe do Dataset
class MNIST(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#$ Classe do modelo MLP
class MLP_Model(nn.Module):
    def __init__(self, camada = 0, neuronio1 = 0, neuronio2 = 0, neuronio3 = 0, fnCusto = 0, fnAtivacao = 0, dropout = 0, lnRate = 0, epoca = 0):
        super(MLP_Model, self).__init__()
        self.X1 = camada
        self.A1 = nn.Flatten()
        self.A2 = fnAtivacao
        self.C1 = fnCusto
        self.Z1 = lnRate
        self.W1 = epoca
        self.D1 = nn.Dropout(dropout)
        self.L1 = nn.Linear(28 * 28, neuronio1)
        self.L2 = nn.Linear(neuronio1, neuronio2)
        self.L3 = nn.Linear(neuronio2, neuronio3)

        # Ajuste na camada final
        if isinstance(fnCusto, nn.CrossEntropyLoss):
            self.L_final = nn.Linear(neuronio1 if camada == 1 else neuronio2 if camada == 2 else neuronio3, 10)  # Saída com 10 neurônios para classificação (10 classes)
        elif isinstance(fnCusto, nn.MSELoss):
            self.L_final = nn.Linear(neuronio1 if camada == 1 else neuronio2 if camada == 2 else neuronio3, 1)  # Saída com 1 neurônio para regressão (exemplo)

    def forward(self, x):
        x = self.A1(x) #* Flatten
        x = self.L1(x) #* Camada de entrada
        if self.X1 == 1:
            x = self.A2(x) #* Função de ativação
        elif self.X1 == 2:
            x = self.A2(x) #* Função de ativação
            x = self.L2(x) #* Camada de saída
        elif self.X1 == 3:
            x = self.A2(x) #* Função de ativação
            x = self.L2(x) #* Camada oculta 1
            x = self.A2(x) #* Função de ativação
            x = self.L3(x) #* Camada de saída

        x = self.L_final(x) #* Saída final
        return x

#$ Função de treinamento
def training(N_Epochs, model, loss_fn, opt):
    loss_list = []

    for epoch in tqdm(range(N_Epochs + 1)):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Para CrossEntropyLoss, os rótulos devem ser inteiros e a saída deve ser logits
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                yb = yb.long()  # Certifique-se de que os rótulos são inteiros
            # Para MSELoss, os rótulos devem ser floats
            elif isinstance(loss_fn, nn.MSELoss):
                yb = yb.float().unsqueeze(1)  # Para MSELoss, os rótulos precisam ser floats
            
            y_pred = model(xb.float())
            loss = loss_fn(y_pred, yb) #* resultado da rede, resultado esperado

            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_list.append(loss.item())

#$ Definindo faixas dos parâmetros
#$$ MLP:
camadas = [1, 2, 3]
neuronios = [20, 40, 80]
#! funcaoCusto = [nn.MSELoss(), nn.CrossEntropyLoss()]
funcaoCusto = [nn.MSELoss()]
#! funcaoCusto = [nn.CrossEntropyLoss()]
funcaoAtivacao = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
dropout = [0, 0.1, 0.3, 0.5]
learningRate = [0.0000000003, 0.00003, 0.03]
epocas = [20, 50, 80, 100]

#$$ CNN:
tamanhoKernel = [2, 4, 8]
strideKernel = [1, 2]
#funcaoCusto
#funcaoAtivacao
convolucoes = [1, 2]
tamanhoKernelPolling = [2, 4]
mapasCaracteristica = [5, 10, 15]

#$ Início do treinamento
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data() #* baixando o banco de dados mnist
train_dl = DataLoader(MNIST(train_X, train_y), batch_size=256, shuffle=True)
test_dl = DataLoader(MNIST(test_X, test_y), batch_size=64)

tempoTotal = 0
count = 1
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for a in camadas:
    for b in funcaoCusto:
        for c in funcaoAtivacao:
            for d in dropout:
                for e in learningRate:
                    for f in epocas:
                        model = MLP_Model(a, neuronios[0], neuronios[1], neuronios[2], b, c, d, e, f)
                        model.to(device)
                        opt = torch.optim.Adam(model.parameters(), lr=model.Z1)
                        inicio = time.time()
                        training(model.W1, model, model.C1, opt)

#$ Avaliação do modelo
                        with torch.no_grad():
                            model.eval()
                            y_pred = []
                            y_true = []

                            for xb, yb in test_dl:
                                xb, yb = xb.to(device), yb.to(device)
                                y_predb = model(xb.float())
                                y_pred.append(y_predb)
                                y_true.append(yb)

                            y_pred = torch.cat(y_pred).to('cpu')
                            y_true = torch.cat(y_true).to('cpu')

                            yf = torch.argmax(y_pred, dim=1)
                            report = classification_report(y_true, yf, output_dict=True, zero_division=0)  # Obter como dicionário

                            final = time.time()
                            tempoExecucao = final - inicio
                            tempoTotal = tempoTotal + tempoExecucao
                            tempoMedio = tempoTotal / count
                            tempoRestante = tempoMedio * (432 - count)
                            horasRestante = tempoRestante // 3600
                            minutosRestante = (tempoRestante % 3600) // 60
                            segundosRestante = tempoRestante % 60

                            # Adicione os resultados e informações do modelo à lista
                            results.append({
                                "Device": next(model.parameters()).device,
                                "Iteration": str(count) + "/432",
                                "Camadas": model.X1,
                                "FnAtivação": model.A2,
                                "FnCusto": model.C1,
                                "LnRate": model.Z1,
                                "Epocas": model.W1,
                                "DropOut": model.D1.p,
                                "Precision": report["weighted avg"]["precision"],
                                "Recall": report["weighted avg"]["recall"],
                                "F1-Score": report["weighted avg"]["f1-score"],
                                "Accuracy": report["accuracy"],
                                "Duração_Execução": format(tempoExecucao, ".4f").replace(".", ","),
                                "Média_Duração": format(tempoMedio, ".4f").replace(".", ","),
                                "Expectativa_Fim": format(horasRestante, ".0f") + "h " + format(minutosRestante, ".0f") + "m " + format(segundosRestante, ".2f") + "s",
                            })

                            print(f"Fim do treinamento {count}")
                            count += 1

                            # Convertendo os resultados para um DataFrame
                            df = pd.DataFrame(results)

                            # Salvando em uma planilha Excel
                            df.to_excel("classification_results.xlsx", index=False, engine='openpyxl')
