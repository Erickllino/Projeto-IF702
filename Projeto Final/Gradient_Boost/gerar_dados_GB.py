import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.exceptions import TrialPruned
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

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
kf = SKFold(n_splits=5, shuffle=True, random_state=50)

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
     # Listas para armazenar as métricas


     val_accuracies = []
     val_ks_values = []
     val_auc_values = []
     val_precision_values = []
     val_recall_values = []
     val_f1_values = []

     for train_index, val_index in kf.split(df_data, df_target):
          X_train, X_val = df_data.iloc[train_index], df_data.iloc[val_index]
          y_train, y_val = df_target.iloc[train_index], df_target.iloc[val_index]
          
          model.fit(X_train, y_train)
          
          # Previsões e probabilidades
          y_val_pred = model.predict(X_val)
          y_val_proba = model.predict_proba(X_val)[:, 1]
          val_acc = accuracy_score(y_val, y_val_pred) 

          # Cálculo do KS (diferença máxima entre TPR e FPR)
          fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
          val_ks = max(np.abs(tpr_val - fpr_val))

          # Cálculo do AUC-ROC
          val_auc = roc_auc_score(y_val, y_val_proba)

          # Cálculo de precision, recall e f1-score         
          val_precision = precision_score(y_val, y_val_pred)          
          val_recall = recall_score(y_val, y_val_pred)         
          val_f1 = f1_score(y_val, y_val_pred) 

          # Armazenando os resultados          
          val_accuracies.append(val_acc)
          val_ks_values.append(val_ks)         
          val_auc_values.append(val_auc)      
          val_precision_values.append(val_precision)        
          val_recall_values.append(val_recall)        
          val_f1_values.append(val_f1)

     
     mean_val_acc = np.mean(val_accuracies)
     mean_val_ks = np.mean(val_ks_values)
     mean_val_precision = np.mean(val_precision_values)
     mean_val_recall = np.mean(val_recall_values)
     mean_val_f1 = np.mean(val_f1_values)
     mean_val_auc = np.mean(val_auc_values)

     val_metrics = {
          'accuracy': mean_val_acc,
          'ks': mean_val_ks,
          'precision': mean_val_precision,
          'recall': mean_val_recall,
          'f1_score': mean_val_f1,
          'auc_roc': mean_val_auc
    }

     for metric_name, metric_value in val_metrics.items():
          trial.set_user_attr(f'val_{metric_name}', float(metric_value))

     # Cálculo das métricas médias nos folds


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
          "val_accuracy": mean_val_acc,
          "val_ks": mean_val_ks
     })

     # Retorna o valor a ser maximizado; neste caso, a média do KS na validação
     return mean_val_ks



# Criar o estudo do Optuna e otimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective_gb, n_trials=500, show_progress_bar=True)

# Obter DataFrame com os resultados dos trials
trials_df = study.trials_dataframe()
trials_df['param_observado'] = 'Maximize KS'

# Nome do arquivo Excel a ser gerado
file_name = 'Projeto Final/Gradient_Boost/optuna_results_GB.xlsx'


# Cria/atualiza o arquivo Excel com duas abas: 'Trials' e 'Training_History'
if Path(file_name).exists():
    # Carregar dados existentes
    existing_df = pd.read_excel(file_name)
    
    # Combinar com novos dados e remover possíveis duplicatas
    combined_df = pd.concat([existing_df, trials_df], ignore_index=True)
    
    # Remover duplicatas baseado no ID do trial (se aplicável)
    if 'trial_id' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['trial_id'], keep='last')
    else:
        combined_df = combined_df.drop_duplicates()
    
    # Salvar dados combinados
    combined_df.to_excel(file_name, index=False)
    print("\nDados novos foram adicionados ao arquivo existente!")
else:
    # Salvar pela primeira vez
    trials_df.to_excel(file_name, index=False)
    print("\nArquivo criado com os resultados dos testes do Optuna!")


