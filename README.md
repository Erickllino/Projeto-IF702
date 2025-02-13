# PROJETOS - IF702 (Introdução à redes Neurais)

# 1o MiniProjeto

## Introdução

Este projeto tem como objetivo explorar e comparar o desempenho de Redes Neurais Multicamadas (MLP) e Redes Neurais Convolucionais (CNN) no reconhecimento de dígitos escritos à mão, utilizando o dataset MNIST. Além disso, realizaremos experimentos com diversas configurações de hiperparâmetros para identificar a arquitetura mais eficiente.

### Discente(s): 
Bruno Antonio dos Santos Bezerra, Erick Vinicius Rebouças Cruz, Gabriel Monteiro Silva

---

## Objetivos

1. Implementar e treinar modelos MLP e CNN.
2. Analisar e comparar o desempenho dos modelos.
3. Explorar variações de hiperparâmetros para otimização de desempenho.

---

## Instruções e Requisitos

### Tarefas

#### **Implementação :**

- **MLP:**
  - Modelo com:
    - Uma camada de entrada (784 neurônios).
    - Camadas ocultas configuráveis.
    - Uma camada de saída (10 neurônios, uma para cada classe do MNIST).
  - Suporte para funções de ativação como ReLU, Sigmoid e tanh.
  - Implementação de técnicas como forward e backpropagation, e otimização (ex.: SGD ou Adam).

- **CNN:**
  - Modelo com:
    - Camadas convolucionais e de pooling.
    - Dropout para regularização.
    - Camadas totalmente conectadas para saída.
  - Treinamento e avaliação nos mesmos parâmetros do MLP.

#### **Treinamento e Avaliação:**
- Treinamento utilizando o dataset MNIST.
- Avaliação do desempenho em termos de:
  - Acurácia.
  - Matriz de confusão.
  - Curva de perda e validação.

#### **Experimentação:**
- Testes com variações nos seguintes hiperparâmetros:
  - No MLP:

| Hiperparâmetro          | Valores testados                          |
|-------------------------|-------------------------------------------|
| Número de camadas ocultas | 1, 2, 3                                  |
| Neurônios por camada    | 20, 40, 80                                |
| Função de custo         | MSE, Cross Entropy                        |
| Função de ativação      | ReLU, Sigmoid, tanh                       |
| Dropout                | 0, 0.1, 0.3, 0.5                          |
| Learning rate          | 3×10⁻⁵, 3×10⁻², 3×10⁻¹⁰                   |
| Épocas                 | 20, 50, 80, 100                           |

 - Na CNN:
   
| Hiperparâmetro          | Valores testados                          |
|--------------------------|-------------------------------------------|
| tamanho kernel           | 2, 4, 8                                   |
| stride kernel            | 1, 2                                      |
| Função de custo          | MSE, Cross Entropy                        |
| Função de ativação       | ReLU, Sigmoid, tanh                       |
| nr camadas convolução    | 1, 2                                      |
| Learning rate            | 3×10⁻⁵, 3×10⁻², 3×10⁻¹⁰                   |
| Épocas                   | 20, 50, 80, 100                           |
|tamanho kernel max polling| 2, 4                                      |
|nr mapa caracteristica    | 5, 10, 15                                 |

---

## Dataset

- **Principal:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## Requisitos para Entrega

1. Código: 
   - Um Jupyter Notebook ou script Python contendo a implementação dos modelos e variações.
   - Visualizações das curvas de perda, métricas de avaliação e resultados experimentais.

2. Relatório: 
   - Análise dos experimentos realizados, incluindo:
     - Impacto de diferentes configurações de hiperparâmetros no desempenho dos modelos.
     - Comparação de desempenho entre MLP e CNN.

3. Documentação: 
   - Este README.md atualizado com instruções claras e completas.

---

## Dependências

- [Python](https://www.python.org/) (>= 3.8)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)

Para instalar as dependências, utilize:
```bash
pip install -r requirements.txt
```

# 2o MiniProjeto

## Introdução

Este projeto tem como objetivo desenvolver um modelo preditivo para o preço do Bitcoin utilizando uma rede neural recorrente com arquitetura LSTM. Para isso, serão exploradas e implementadas técnicas de pré-processamento de dados, otimização de hiperparâmetros com o Optuna e criação de um modelo LSTM utilizando o framework TensorFlow/Keras. A proposta é analisar a performance do modelo na previsão dos preços futuros a partir de dados históricos, bem como visualizar e interpretar os resultados obtidos.

### Discente(s):
- Bruno Antonio dos Santos Bezerra  
- Erick Vinicius Rebouças Cruz  
- Gabriel Monteiro Lobão

---

## Objetivos

1. Carregar e explorar o dataset de dados históricos do Bitcoin.
2. Realizar o pré-processamento dos dados, incluindo:
   - Conversão de dados de 1 minuto para dados diários.
   - Normalização e criação de sequências temporais (janela de 60 dias).
3. Implementar a otimização dos hiperparâmetros do modelo LSTM utilizando o Optuna.
4. Desenvolver e treinar um modelo LSTM, validando-o com Early Stopping.
5. Avaliar o desempenho do modelo por meio de métricas (por exemplo, R² score).
6. Realizar previsões no conjunto de teste e projetar preços futuros (exemplo para os próximos 30 dias).
7. Visualizar os resultados através de gráficos comparativos entre dados reais e previsões.

---

## Instruções e Requisitos

### 1. Carregamento e Exploração dos Dados

- **Dataset:**
  - Fonte: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data).
  - Realizar a importação do arquivo CSV contendo dados em intervalo de 1 minuto.
- **Conversão:**
  - Converter o timestamp para o formato datetime.
  - Agregar os dados para obter informações diárias (usando funções como `resample` para obter o primeiro valor de abertura, máximo, mínimo, último valor de fechamento e volume total).

### 2. Pré-processamento

- Selecionar a coluna **Close** para previsão.
- Normalizar os dados utilizando o **MinMaxScaler**.
- Dividir os dados em conjuntos de treino e teste (80% para treino e 20% para teste).
- Criar sequências temporais (janela de 60 dias) para alimentar a LSTM.
- Realizar o reshape dos dados para a formatação exigida pela rede LSTM (formato: `[amostras, sequência, features]`).

### 3. Otimização de Hiperparâmetros com Optuna

- Implementar uma função `objective` que possibilite:
  - Definir o número de camadas LSTM (entre 1 e 3).
  - Sugerir o número de unidades para cada camada LSTM (ex.: de 25 a 200, com incremento de 10).
  - Sugerir o dropout rate para cada camada (valores de 0.1 a 0.5).
  - Sugerir o número de unidades para uma camada densa intermediária (ex.: de 10 a 100, com incremento de 10).
  - Ajustar hiperparâmetros como **batch_size** (entre 16 e 128) e **learning_rate** (entre 1e-5 e 1e-2, escala logarítmica).
- Executar um total de 250 trials e registrar os resultados, utilizando como métrica principal a *validation loss* e a precisão medida pelo **R² score** no conjunto de teste.
- Exportar os resultados dos trials para um arquivo Excel (ex.: `optuna_results.xlsx`).

> **Tabela de Hiperparâmetros Testados:**

| Hiperparâmetro          | Valores testados                                  |
|-------------------------|---------------------------------------------------|
| Número de camadas LSTM  | 1, 2, 3                                           |
| Unidades LSTM           | 25, 35, 45, …, 200 (passo de 10)                    |
| Dropout Rate            | 0.1, 0.2, 0.3, 0.4, 0.5                             |
| Unidades da camada densa| 10, 20, 30, …, 100 (passo de 10)                    |
| Batch Size              | 16, 32, 48, …, 128                                 |
| Learning Rate           | [1e-5, 1e-2] (log scale)                           |


### 4. Criação do Modelo LSTM

- Desenvolver o modelo final com base nos melhores hiperparâmetros encontrados na etapa de otimização.
- Exemplo de arquitetura final:
  - Camada LSTM com 165 unidades (sem retorno de sequência).
  - Camada de Dropout (por exemplo, 0.1).
  - Camada densa com 60 unidades e função de ativação *ReLU*.
  - Camada de saída com 1 unidade para prever o preço.

### 5. Treinamento e Validação do Modelo

- Compilar o modelo utilizando o otimizador **Adam** e a função de perda **mean_squared_error**.
- Treinar o modelo com:
  - Um número elevado de épocas (ex.: 300 épocas).
  - Batch size fixo (ex.: 16).
  - Validação interna (validation_split de 20%).
  - Uso do **EarlyStopping** para prevenir overfitting, monitorando a *val_loss*.

### 6. Avaliação e Previsões

- Realizar previsões para os conjuntos de treino e teste.
- Reverter a normalização dos dados para que os valores previstos possam ser comparados com os dados reais.
- Plotar gráficos comparativos:
  - Previsões versus preços reais.
  - Curvas de perda (treino e validação).

### 7. Previsão Futura

- Desenvolver uma função para realizar previsões futuras (ex.: para os próximos 30 dias), utilizando as últimas 60 amostras conhecidas.
- Desnormalizar e plotar as previsões junto com o histórico dos preços, possibilitando a comparação visual.

---

## Dataset

- **Principal:** [Bitcoin Historical Data - Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## Requisitos para Entrega

1. **Código:**
   - Um Jupyter Notebook ou script Python contendo a implementação completa do projeto, incluindo:
     - Carregamento e exploração dos dados.
     - Pré-processamento e criação de sequências para a LSTM.
     - Otimização de hiperparâmetros com Optuna.
     - Construção, treinamento e validação do modelo LSTM.
     - Geração de gráficos e visualizações dos resultados.
2. **Relatório:**
   - Análise detalhada dos experimentos realizados, incluindo:
     - Descrição dos passos de pré-processamento.
     - Impacto dos diferentes hiperparâmetros na performance do modelo.
     - Discussão dos resultados obtidos (validação loss e R² score).
     - Interpretação dos gráficos de comparação entre valores reais e previstos, bem como as previsões futuras.
3. **Documentação:**
   - Atualização deste README.md com instruções claras de execução e dependências utilizadas.

---

## Dependências

- [Python 3.9](https://www.python.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)  
  *Recomendado instalar via:*
  ```bash
  conda install -c conda-forge tensorflow-gpu=2.10


