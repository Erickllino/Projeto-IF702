import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from KAN.tabkanet.models import TabMLPNet
from tabkanet.tools import seed_everything, get_dataset, get_data_loader, train
from tabkanet.metrics import f1_score_macro

# Fixar a semente para reprodutibilidade
seed = 0
seed_everything(seed)

def load_breast_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    
    # Normalizar os dados
    scaler = StandardScaler()
    df[data.feature_names] = scaler.fit_transform(df[data.feature_names])
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        df.drop(columns=["target"]), df["target"], test_size=0.3, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Função para calcular os bins
def get_quantile_bins(x_cont, n_bins=4):
    feature_dim = x_cont.shape[1]
    bins = torch.zeros(feature_dim, n_bins + 1)
    for i in range(feature_dim):
        # Converta a coluna específica para tensor e depois calcule os quantis
        quantiles = torch.quantile(torch.tensor(x_cont.iloc[:, i].values, dtype=torch.float32), torch.linspace(0, 1, n_bins + 1))
        bins[i] = quantiles
    return bins


# Carregar o dataset Breast Cancer
X_train, X_val, X_test, y_train, y_val, y_test = load_breast_cancer_data()

# Definir features
continuous_features = list(X_train.columns)
categorical_features = []
target_name = "target"
task = "classification"

# Criar datasets
df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

dataset_train, dataset_val, dataset_test = get_dataset(
    df_train, df_val, df_test, target_name, task, categorical_features, continuous_features
)

dataloader_train, dataloader_val, dataloader_test = get_data_loader(
    dataset_train, dataset_val, dataset_test, train_batch_size=32, inference_batch_size=32
)

# Calcular bins
bins = get_quantile_bins(X_train)

# Definir o modelo
model = TabMLPNet(
    output_dim=2,  # 2 classes (binário)
    vocabulary={},
    num_continuous_features=len(continuous_features),
    embedding_dim=16, 
    mlp_hidden_dims=[32],
    activation="relu",
    ffn_dropout_rate=0.1,
    nhead=8,
    num_layers=3,
    dim_feedforward=128,
    attn_dropout_rate=0.1,
    learninable_noise=True,
    bins=bins
)

# Definir otimizador e loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

# Treinar o modelo
train_history, val_history, test_history = train(
    model, epochs=10, task=task, train_loader=dataloader_train, val_loader=dataloader_val,
    test_loader=dataloader_test, optimizer=optimizer, criterion=criterion,
    custom_metric=f1_score_macro, maximize=False, early_stopping_patience=5, gpu_num=0
)