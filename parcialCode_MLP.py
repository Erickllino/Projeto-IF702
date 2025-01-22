
#$ Bibliotecas Básicas
import matplotlib.pyplot as plt #* para plotar gráficos
from tqdm import tqdm #* para criar barras de progresso 
from sklearn.metrics import classification_report #* para obter dados estatíticos de eficiência dos resultados
import keras #* para obter o banco de dados necessário para o treinamento
import pandas as pd
#! import pandas as pd #*
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

    def forward(self, x):
        if self.X1 == 1:
            x = self.A1(x) #* Flatten
            x = self.L1(x) #* Camada de entrada/saída
            x = self.A2(x) #* Função de ativação
        elif self.X1 == 2:
            x = self.A1(x) #* Flatten
            x = self.L1(x) #* Camada de entrada
            x = self.A2(x) #* Função de ativação
            x = self.L2(x) #* Camada de saída
        elif self.X1 == 3:
            x = self.A1(x) #* Flatten
            x = self.L1(x) #* Camada de entrada
            x = self.A2(x) #* Função de ativação
            x = self.L2(x) #* Camada óculta 1
            x = self.A2(x) #* Função de ativação
            x = self.L3(x) #* Camada de saída
            
        return x

#$ Função de treinamento
def training(N_Epochs, model, loss_fn, opt):
    loss_list = []

    for epoch in tqdm(range(N_Epochs + 1)):
        for xb, yb in train_dl:
            y_pred = model(xb.float())
            loss = loss_fn(y_pred, yb.long())

            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_list.append(loss.item())

    #! plt.title("Cost Decay")
    #! plt.plot(loss_list)
    #! plt.xlabel("Epoch")
    #! plt.ylabel("Cost")
    #! plt.show()
    #! plt.figure(figsize=(14, 6))

#$ Definindo faixas dos parâmetros
#$$ MLP:
camadas = [1, 2, 3]
neuronios = [20, 40, 80]
funcaoCusto = [nn.CrossEntropyLoss(), nn.MSELoss()]
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

count = 1
results = []
for a in camadas:
    for b in funcaoCusto:
        for c in funcaoAtivacao:
            for d in dropout:
                for e in learningRate:
                    for f in epocas:
                        model = MLP_Model(a, neuronios[0], neuronios[1], neuronios[2], b, c, d, e, f)
                        opt = torch.optim.Adam(model.parameters(), lr=model.Z1)
                        training(model.W1, model, model.C1, opt)

#$ Avaliação do modelo
                        with torch.no_grad():
                            model.eval()
                            y_pred = []
                            y_true = []

                            for xb, yb in test_dl:
                                y_predb = model(xb.float())
                                y_pred.append(y_predb)
                                y_true.append(yb)

                            y_pred = torch.cat(y_pred)
                            y_true = torch.cat(y_true)

                            yf = torch.argmax(y_pred, dim=1)
                            report = classification_report(y_true, yf, output_dict=True)  # Obter como dicionário

                            # Adicione os resultados e informações do modelo à lista
                            results.append({
                                "Iteration": count,
                                "Camadas": model.X1,
                                "FnAtivação": model.A2,
                                "FnCusto": model.C1,
                                "LnRate": model.Z1,
                                "Epocas": model.W1,
                                "DropOut": model.D1,
                                "Precision": report["weighted avg"]["precision"],
                                "Recall": report["weighted avg"]["recall"],
                                "F1-Score": report["weighted avg"]["f1-score"],
                                "Accuracy": report["accuracy"],
                            })

                            print(f"Fim do treinamento {count}")
                            count += 1

                            # Convertendo os resultados para um DataFrame
                            df = pd.DataFrame(results)

                            # Salvando em uma planilha Excel
                            df.to_excel("classification_results.xlsx", index=False, engine='openpyxl')

















                            
                            # print(classification_report(y_true, yf))
                            # print()
                            # print("Fim do treinamento", count)
                            # print("Dados da RN:")
                            # print("Camada:", model.X1)
                            # print("FnAtivação:", model.A2)
                            # print("FnCusto:", model.C1)
                            # print("LnRate:", model.Z1)
                            # print("Epoca:", model.W1)
                            # print("DropOut:", model.D1)
                            # count = count + 1
                            # print("-----------------------------------------------------")
                            # print()

# # Carregar dados experimentais e visualização
# df = pd.read_json('Experimentos_MLP.json')

# sns.scatterplot(data=df, x="epochs", y="precision")
# sns.lineplot(data=df, x="epochs", y="precision", color='orange')
# sns.boxplot(data=df, x="learning_rate", y="precision")
# sns.pairplot(df, vars=["epochs", "learning_rate", "precision"])
# sns.pairplot(df, vars=["epochs", "learning_rate", "precision"], hue="shuffle")

# pivot_table = df.pivot_table(values="precision", index="epochs", columns="learning_rate")
# sns.heatmap(pivot_table, annot=True, cmap="coolwarm")

# from scipy.stats import linregress

# sns.scatterplot(data=df, x="epochs", y="precision")
# sns.regplot(data=df, x="epochs", y="precision", scatter=False, color="orange")
# plt.title("Efeito de Epochs na Precisão")
# plt.show()

# slope, intercept, r_value, p_value, std_err = linregress(df['epochs'], df['precision'])
# print(f"A precisão varia {slope:.2f} por época. R²: {r_value**2:.2f}")

# sns.boxplot(data=df, x="learning_rate", y="precision")
# plt.title("Efeito de Learning Rate na Precisão")
# plt.show()
# print(df.groupby("learning_rate")["precision"].mean())

# sns.boxplot(data=df, x="shuffle", y="precision")
# plt.title("Efeito de Shuffle na Precisão")
# plt.show()
# shuffle_mean = df[df["shuffle"] == True]["precision"].mean()
# no_shuffle_mean = df[df["shuffle"] == False]["precision"].mean()
# print(f"O shuffle aumenta a precisão em {shuffle_mean - no_shuffle_mean:.2f}")

# # Atualização dos tensores de entrada
# train_X = train_X[:, None, :, :]
# test_X = test_X[:, None, :, :]

# train_dl = DataLoader(MNIST(train_X, train_y), batch_size=256, shuffle=True)
# test_dl = DataLoader(MNIST(test_X, test_y), batch_size=64)

# # Classe do modelo CNN
# class CNN_Model(nn.Module):
#     def __init__(self):
#         super(CNN_Model, self).__init__()
#         self.Fl = nn.Flatten()
#         self.C1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=4, stride=1)
#         self.P1 = nn.MaxPool2d(kernel_size=2)
#         self.C2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=4, stride=1)
#         self.P2 = nn.MaxPool2d(kernel_size=4)
#         self.L1 = nn.Linear(10 * 2 * 2, 100)
#         self.L2 = nn.Linear(100, 50)
#         self.L3 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = torch.relu(self.C1(x))
#         x = self.P1(x)
#         x = torch.relu(self.C2(x))
#         x = self.P2(x)
#         x = self.Fl(x)
#         x = torch.relu(self.L1(x))
#         x = torch.relu(self.L2(x))
#         x = self.L3(x)
#         return x

# # Treinamento do modelo CNN
# epoch = 50
# model = CNN_Model()
# loss = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.00003)
# training(epoch, model, loss, opt)

# # Avaliação do modelo CNN
# with torch.no_grad():
#     model.eval()
#     y_pred = []
#     y_true = []

#     for xb, yb in test_dl:
#         y_predb = model(xb.float())
#         y_pred.append(y_predb)
#         y_true.append(yb)

#     y_pred = torch.cat(y_pred)
#     y_true = torch.cat(y_true)

#     yf = torch.argmax(y_pred, dim=1)
#     print(classification_report(y_true, yf))
