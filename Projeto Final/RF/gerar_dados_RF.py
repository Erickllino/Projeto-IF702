import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.exceptions import TrialPruned
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold

# Imports do Keras (não utilizados para o modelo RF, mas presentes no código original)
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Precision, Recall

# Leitura e preparação dos dados
df = pd.read_csv('Projeto Final/data/customer_churn_telecom_services.csv')

df.drop_duplicates(df.columns, ignore_index=True)
df = df.fillna(0)
#df.dropna(ignore_index=True)
df = pd.get_dummies(df, drop_first=True)

df_data = df.drop(columns='Churn_Yes')
df_target = df['Churn_Yes']



data = df


# Função para dividir os dados em folds (treino, validação e teste)
def k_fold_train_val_test(data, k=5, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_val_indices = indices[n_test:]
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_indices = []
    for train_idx, val_idx in kf.split(train_val_indices):
        train_indices = train_val_indices[train_idx]
        val_indices = train_val_indices[val_idx]
        fold_indices.append((train_indices, val_indices, test_indices))
    return fold_indices

folds = k_fold_train_val_test(data, k=5, test_size=0.25, random_state=42)

# Escolha do fold (altere fold_order para 0, 1, 2, 3 ou 4 conforme necessário)
fold_order = 0
df_treino = data.iloc[folds[fold_order][0]]
df_test = data.iloc[folds[fold_order][1]]
df_validacao = data.iloc[folds[fold_order][2]]

# Separando as features e o target
X_treino = df_treino.drop(columns=['Churn_Yes'])
y_treino = df_treino['Churn_Yes']
X_test = df_test.drop(columns=['Churn_Yes'])
y_test = df_test['Churn_Yes']
X_val = df_validacao.drop(columns=['Churn_Yes'])
y_val = df_validacao['Churn_Yes']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_treino)
y_train = y_treino
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]

# Função para calcular a estatística KS
def ks_statistic(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)

# Função para calcular métricas adicionais no conjunto de validação
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return {"auc": auc, "precision": precision, "recall": recall}

# Lista para armazenar o histórico de treinamento (épocas) de todos os trials
training_history = []

def objective(trial):
    # Sugestão de hiperparâmetros
    max_depth = trial.suggest_int("max_depth", 2, 16)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    total_estimators = trial.suggest_int("total_estimators", 50, 200)
    
    step = 10  # Incremento de árvores por "época"
    
    # Criação do modelo com warm_start para adicionar árvores progressivamente
    model = RandomForestClassifier(
        n_estimators=0,   # Inicia sem árvores
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )
    
    best_val_score = 0
    epoch = 0  # contador de épocas para reportar ao Optuna
    
    # Loop para treinamento progressivo
    for n in range(step, total_estimators + 1, step):
        epoch += 1
        model.n_estimators = n
        model.fit(X_train, y_train)
        
        # Cálculo das predições e probabilidades
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        
        # Cálculo das métricas
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        train_ks = ks_statistic(y_train, train_prob)
        val_ks = ks_statistic(y_val, val_prob)
        
        # Armazena os resultados da iteração atual
        training_history.append({
            "trial": trial.number,
            "epoch": epoch,
            "n_estimators": n,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_ks": train_ks,
            "val_ks": val_ks
        })
        
        # Calcula métricas adicionais no conjunto de validação
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Reporta a acurácia de validação com o contador 'epoch' para evitar duplicação do step
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise TrialPruned()
        
        # Armazena as métricas adicionais como atributos do trial
        for metric_name, metric_value in val_metrics.items():
            trial.set_user_attr(f'val_{metric_name}', float(metric_value))
        
        # O objetivo é maximizar a estatística KS de validação
        best_val_score = max(best_val_score, val_ks)
    
    return best_val_score

# Criação do estudo Optuna (maximizando e utilizando o pruner MedianPruner)
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=500)

# Obtém o dataframe com os trials do Optuna
trials_df = study.trials_dataframe()
trials_df['param_observado'] = 'Maximize KS'

# Nome do arquivo Excel a ser gerado
file_name = 'Projeto Final/RF/optuna_results_RF.xlsx'

# Cria/atualiza o arquivo Excel com duas abas: 'Trials' e 'Training_History'
if Path(file_name).exists():
    # Carrega os dados existentes (todas as abas)
    existing_data = pd.read_excel(file_name, sheet_name=None)
    trials_existing = existing_data.get('Trials', pd.DataFrame())
    history_existing = existing_data.get('Training_History', pd.DataFrame())
    
    new_trials_df = trials_df.copy()
    new_history_df = pd.DataFrame(training_history)
    
    # Combina os dados e remove duplicatas
    if not trials_existing.empty:
         trials_combined = pd.concat([trials_existing, new_trials_df], ignore_index=True).drop_duplicates(subset=['trial_number'], keep='last')
    else:
         trials_combined = new_trials_df
    if not history_existing.empty:
         history_combined = pd.concat([history_existing, new_history_df], ignore_index=True).drop_duplicates()
    else:
         history_combined = new_history_df

    with pd.ExcelWriter(file_name, engine='openpyxl', mode='w') as writer:
         trials_combined.to_excel(writer, sheet_name='Trials', index=False)
         history_combined.to_excel(writer, sheet_name='Training_History', index=False)
    print("\nNovos dados adicionados ao arquivo existente!")
else:
    new_trials_df = trials_df.copy()
    new_trials_df['param_observado'] = 'Maximize KS'
    new_history_df = pd.DataFrame(training_history)
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
         new_trials_df.to_excel(writer, sheet_name='Trials', index=False)
         new_history_df.to_excel(writer, sheet_name='Training_History', index=False)
    print("\nArquivo criado com os resultados dos testes do Optuna!")

# Exibe informações do melhor trial encontrado
print("Melhor trial:")
best_trial = study.best_trial
print(f"  Valor: {best_trial.value}")
print("  Parâmetros:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
