# PROJETOS - IF702 (Introdução à redes Neurais)

# 1o MiniProjeto

## Introdução

Este projeto tem como objetivo explorar e comparar o desempenho de Redes Neurais Multicamadas (MLP) e Redes Neurais Convolucionais (CNN) no reconhecimento de dígitos escritos à mão, utilizando o dataset MNIST. Além disso, realizaremos experimentos com diversas configurações de hiperparâmetros para identificar a arquitetura mais eficiente.

### Discente(s): 
Bruno Antonio dos Santos Bezerra, Erick Vinicius Rebouças Cruz, Gabriel Monteiro Lobão

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

| Hiperparâmetro          | Valores testados                          |
|-------------------------|-------------------------------------------|
| Número de camadas ocultas | 1, 2, 3                                  |
| Neurônios por camada    | 20, 40, 80                                |
| Função de custo         | MSE, Cross Entropy                        |
| Função de ativação      | ReLU, Sigmoid, tanh                       |
| Dropout                | 0, 0.1, 0.3, 0.5                          |
| Learning rate          | 3×10⁻⁵, 3×10⁻², 3×10⁻¹⁰                   |
| Épocas                 | 20, 50, 80, 100                           |

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
