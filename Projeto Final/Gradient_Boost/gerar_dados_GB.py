import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.exceptions import TrialPruned
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold as SKFold

# Imports do Keras (não utilizados para o modelo RF, mas presentes no código original)
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Precision, Recall

# Leitura e preparação dos dados
df = pd.read_csv('Projeto Final/data/customer_churn_telecom_services.csv')

# Corrige a remoção de duplicatas (atribui o resultado de volta ao DataFrame)
df = df.drop_duplicates(ignore_index=True)
df = df.fillna(0)
# df.dropna(ignore_index=True)  # opção alternativa, se necessário
df = pd.get_dummies(df, drop_first=True)

df_data = df.drop(columns='Churn_Yes')
df_target = df['Churn_Yes']

# Definir validação cruzada
kf = SKFold(n_splits=3, shuffle=True, random_state=50)

# Lista global para registrar o histórico de treinamento de cada trial
training_history = []

def objective_gb(trial):
    # Hiperparâmetros a serem otimizados para o GradientBoostingClassifier
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    subsample = trial.suggest_float("subsample", 0.3, 1.0)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    
    # Criar o modelo com os hiperparâmetros sugeridos
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=50
    )
    
    # Inicializa listas para armazenar as métricas de cada fold
    train_accuracies = []
    val_accuracies = []
    train_ks_values = []
    val_ks_values = []
    
    for train_index, val_index in kf.split(df_data, df_target):
        X_train, X_val = df_data.iloc[train_index], df_data.iloc[val_index]
        y_train, y_val = df_target.iloc[train_index], df_target.iloc[val_index]
        
        model.fit(X_train, y_train)
        
        # Previsões e probabilidades
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Cálculo das acurácias
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Cálculo do KS (diferença máxima entre TPR e FPR)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
        train_ks = max(np.abs(tpr_train - fpr_train))
        val_ks = max(np.abs(tpr_val - fpr_val))
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_ks_values.append(train_ks)
        val_ks_values.append(val_ks)
    
    # Cálculo das métricas médias nos folds
    mean_train_acc = np.mean(train_accuracies)
    mean_val_acc = np.mean(val_accuracies)
    mean_train_ks = np.mean(train_ks_values)
    mean_val_ks = np.mean(val_ks_values)
    
    # Como "epoch" não é aplicável a este modelo, usaremos o valor 1
    epoch = 1
    
    # Registra os resultados do trial no histórico
    training_history.append({
        "trial": trial.number,
        "epoch": epoch,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "train_accuracy": mean_train_acc,
        "val_accuracy": mean_val_acc,
        "train_ks": mean_train_ks,
        "val_ks": mean_val_ks
    })
    
    # Retorna o valor a ser maximizado; neste caso, a média do KS na validação
    return mean_val_ks

# Criar o estudo do Optuna e otimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective_gb, n_trials=10, show_progress_bar=True)

# Obter DataFrame com os resultados dos trials
trials_df = study.trials_dataframe()
trials_df['param_observado'] = 'Maximize KS'

# Nome do arquivo Excel a ser gerado
file_name = 'Projeto Final/Gradient_Boost/optuna_results_GB.xlsx'

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


