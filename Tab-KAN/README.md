## 🚀 TabKANet: Modelagem de Dados Tabulares com Kolmogorov-Arnold Network e Transformer

Os dados tabulares são amplamente utilizados em diversas áreas, mas redes neurais ainda enfrentam desafios ao lidar com esse tipo de dado, onde modelos como GBDT (Gradient Boosted Decision Trees) geralmente dominam. No entanto, novas arquiteturas estão mudando esse cenário, e é aqui que entra o TabKANet: um modelo inovador que combina KAN (Kolmogorov-Arnold Network) e Transformer para melhor modelagem de dados tabulares.

Principais destaques do modelo:

✅ Módulo de Embedding Numérico baseado em KAN: melhora a representação de features numéricas.

✅ Arquitetura baseada em Transformer: permite capturar relações complexas nos dados.

✅ Facilidade de implementação e estabilidade: desempenho validado em diversos benchmarks públicos.


O modelo foi testado em bases públicas e demonstrou desempenho competitivo!
Caso tenham alguma dúvida para executar esse script com a base de dados do projeto, podem consultar, principalmente, as duas fontes abaixo:

🔗 [Link para o artigo](https://arxiv.org/abs/2409.08806)

🔗 [Link para o código-fonte original](https://github.com/AI-thpremed/TabKANet/tree/main?tab=readme-ov-file#reference)

## 📂 Estrutura do Projeto

📁 Model: A pasta TabKANet contém todos os arquivos relacionados ao modelo.

📁 train_tutorial: Notebook Jupyter com um exemplo de treinamento para um dataset público. A estrutura pode ser facilmente adaptada para outros conjuntos de dados!

## 📦 Dependências

Este projeto utiliza Torch para treinamento e avaliação.

