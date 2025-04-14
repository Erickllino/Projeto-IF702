import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from kan import KAN, KANLayer

# Carregar e pré-processar dados
path_file = 'data/customer_churn_telecom_services.csv'
df = pd.read_csv(path_file)

# 1. Primeiro aplicar o OneHotEncoder nas colunas categóricas
colunas = ['PaymentMethod', 'Contract', 'InternetService']
ohe = OneHotEncoder(dtype=int, drop='if_binary')
colunas_ohe = ohe.fit_transform(df[colunas]).toarray()

# Criar DataFrame com as colunas codificadas
df_ohe = pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(colunas))

# 2. Concatenar com o DataFrame original (removendo as colunas originais)
data = pd.concat([df.drop(colunas, axis=1), df_ohe], axis=1)

# 3. Agora fazer as substituições nos dados combinados
replace_dict = {
    'Yes': 1,
    'No': 0,
    'Female': 1,
    'Male': 0,
    'No internet service': 0,
    'No phone service': -1
}

data.replace(replace_dict, inplace=True)

# 4. Converter todas as colunas para float32
data = data.astype(np.float32).fillna(0)

# Verificar os tipos de dados
print(data.dtypes)

# Preparar dados
X = data.drop('Churn', axis=1).values
y = data['Churn'].values

# Classe Dataset
class ChurnDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, optimizer, criterion, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs).squeeze())
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs > 0.5).float().cpu().numpy())
            y_proba.extend(outputs.cpu().numpy())

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_proba),
    }

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hiperparâmetros
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 4, 12, step=4)
    grid_size = trial.suggest_int('grid_size', 5, 10)
    spline_order = trial.suggest_int('spline_order', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    metrics = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Divisão e normalização
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Datasets
        train_dataset = ChurnDataset(X_train, y_train)
        val_dataset = ChurnDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Modelo KAN
        width = [X.shape[1]] + [hidden_dim] * num_layers + [1]

        model = KAN(
            width=width,
            grid=grid_size,
            k=spline_order,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Treinamento
        train_model(model, train_loader, optimizer, criterion, device)

        # Avaliação
        fold_metrics = evaluate_model(model, val_loader, device)
        metrics.append(fold_metrics)

    # Calcular médias
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    for k, v in avg_metrics.items():
        trial.set_user_attr(k, float(v))

    return avg_metrics['auc_roc']

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Salvar resultados
    results_df = study.trials_dataframe()
    results_df['param_observado'] = 'Maximize AUC'

    file_name = 'Projeto Final/KAN/optuna_results_kan.xlsx'
    Path('data/').mkdir(parents=True, exist_ok=True)

    if Path(file_name).exists():
        existing_df = pd.read_excel(file_name)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True).drop_duplicates()
        combined_df.to_excel(file_name, index=False)
    else:
        results_df.to_excel(file_name, index=False)

    print("\nMelhores hiperparâmetros:")
    print(study.best_params)
    print("\nMétricas médias:")
    print({k: v for k, v in study.best_trial.user_attrs.items()})