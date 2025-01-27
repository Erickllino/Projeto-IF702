import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados da planilha (supondo um arquivo .csv)
df = pd.read_excel('MLP_data complete 1977246.xlsx.xlsx')

# Definir o estilo do gráfico
sns.set(style="whitegrid")

# Função para plotar gráficos
def plot_graphs(df):
    # # 1. Gráfico de linha: Precisão vs Número de Épocas
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='Epocas', y='Precision', data=df, marker='o')
    # plt.title('Precisão vs. Número de Épocas')
    # plt.xlabel('Número de Épocas')
    # plt.ylabel('Precisão')
    # plt.show()

    # # 2. Gráfico de dispersão: Recall vs Acurácia
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='Recall', y='Accuracy', data=df, hue='FnAtivação', style='FnCusto', palette="Set1")
    # plt.title('Recall vs. Acurácia')
    # plt.xlabel('Recall')
    # plt.ylabel('Acurácia')
    # plt.legend(title='Funções', loc='upper left')
    # plt.show()

    # 3. Gráfico de barras: Função de Custo por Função de Ativação
    plt.figure(figsize=(10, 6))
    sns.barplot(x='FnCusto', y='Accuracy', hue='FnAtivação', data=df)
    plt.title('Função de Custo vs. Accuracy por Função de Ativação')
    plt.xlabel('Função de Custo')
    plt.ylabel('Accuracy')
    plt.show()

    # # 4. Gráfico de histograma: Distribuição da Taxa de Aprendizado
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df['LnRate'], kde=True, bins=15)
    # plt.title('Distribuição da Taxa de Aprendizado (LnRate)')
    # plt.xlabel('Taxa de Aprendizado (LnRate)')
    # plt.ylabel('Frequência')
    # plt.show()

    # # 5. Gráfico de caixa (boxplot): Comparação entre DropOut e F1-Score
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='DropOut', y='F1-Score', data=df)
    # plt.title('Distribuição de F1-Score por DropOut')
    # plt.xlabel('DropOut')
    # plt.ylabel('F1-Score')
    # plt.show()

    # # 6. Gráfico de violino: Acurácia por Função de Custo e Função de Ativação
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(x='FnCusto', y='Accuracy', hue='FnAtivação', split=True, data=df)
    # plt.title('Acurácia por Função de Custo e Função de Ativação')
    # plt.xlabel('Função de Custo')
    # plt.ylabel('Acurácia')
    # plt.show()

# Chamar a função para gerar os gráficos
plot_graphs(df)
