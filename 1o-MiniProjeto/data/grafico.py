import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar a planilha
file_path = "MLP_data complete 1977246.xlsx.xlsx"  # Atualize com o caminho do arquivo
df = pd.read_excel(file_path)

# Selecionar as colunas relevantes
relevant_columns = ['Camadas', 'FnAtivação', 'FnCusto', 'LnRate', 'Epocas', 'DropOut', 'Accuracy']
df_filtered = df[relevant_columns].dropna()

# Garantir que colunas categóricas sejam tratadas como strings
categorical_columns = ['Camadas', 'FnAtivação', 'FnCusto']
for col in categorical_columns:
    df_filtered[col] = df_filtered[col].astype(str)

# # Gráfico 1: Boxplot - Impacto de DropOut na Acurácia
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='DropOut', y='Accuracy', data=df_filtered, palette="viridis")
# plt.title('Impacto do DropOut na Acurácia', fontsize=14)
# plt.xlabel('Taxa de DropOut', fontsize=12)
# plt.ylabel('Acurácia', fontsize=12)
# plt.show()

# Gráfico 2: Scatterplot - Número de Épocas vs Acurácia
plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Epocas', y='Accuracy', hue='FnAtivação', data=df_filtered, palette="deep")
sns.violinplot(x='Epocas', y='Accuracy', hue='FnAtivação', data=df_filtered, split="deep")
plt.title('Impacto do Número de Épocas na Acurácia', fontsize=14)
plt.xlabel('Número de Épocas', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.legend(title='Função de Ativação')
plt.show()

# # Gráfico 3: Barplot - Acurácia por Função de Custo e Camadas
# plt.figure(figsize=(14, 7))
# sns.barplot(
#     x='FnCusto', y='Accuracy', hue='Camadas', data=df_filtered, ci=None, palette="coolwarm"
# )
# plt.title('Acurácia por Função de Custo e Arquitetura de Camadas', fontsize=14)
# plt.xlabel('Função de Custo', fontsize=12)
# plt.ylabel('Acurácia', fontsize=12)
# plt.legend(title='Arquitetura de Camadas')
# plt.show()

# # Gráfico 4: Heatmap - Correlação entre Variáveis e Acurácia
# correlation_columns = ['LnRate', 'Epocas', 'DropOut', 'Accuracy']
# correlation_data = df_filtered[correlation_columns].corr()

# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_data, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title('Correlação entre Variáveis e Acurácia', fontsize=14)
# plt.show()
