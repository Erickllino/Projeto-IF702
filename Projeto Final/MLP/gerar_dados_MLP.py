import numpy as np
import pandas as pd
import optuna
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Precision, Recall
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score





df = pd.read_csv('Projeto Final/data/customer_churn_telecom_services.csv')

df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']] = df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].replace({'Yes': 1, 'No': 0})
df[['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] = df[['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df['gender'] = df['gender'].replace({'Female':1, 'Male':0})
df['MultipleLines'] =  df['MultipleLines'].replace({'No phone service': -1})


colunas = ['PaymentMethod', 'Contract', 'InternetService']
ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(df[colunas]).toarray()
dados = pd.concat([df, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(colunas))], axis=1)

data = dados.drop(colunas, axis=1)

data = data.fillna(0)



def k_fold_train_val_test(data, k=5, test_size=0.2, random_state=42):

    # Garante reprodutibilidade
    np.random.seed(random_state)
    
    # Número total de amostras
    n_samples = len(data)
    
    # Separação inicial entre conjunto de treinamento+validação e conjunto de teste
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Número de amostras para teste
    n_test = int(n_samples * test_size)
    
    # Índices para teste (fixos para todos os folds)
    test_indices = indices[:n_test]
    
    # Índices para treinamento + validação
    train_val_indices = indices[n_test:]
    
    # Aplicar k-fold nos índices de treinamento + validação
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Lista para armazenar os índices de cada fold
    fold_indices = []
    
    # Iterar sobre os splits do k-fold
    for train_idx, val_idx in kf.split(train_val_indices):
        # Mapear os índices do split para os índices originais do dataset
        train_indices = train_val_indices[train_idx]
        val_indices = train_val_indices[val_idx]
        
        # Armazenar os índices deste fold
        fold_indices.append((train_indices, val_indices, test_indices))
    
    return fold_indices


folds = k_fold_train_val_test(data, k=5, test_size=0.25, random_state=42)


# Troque o valor de fold_order para 0, 1, 2, 3 ou 4 para escolher um dos folds
fold_order = 0 #
df_treino = data.iloc[folds[fold_order][0]]
df_test = data.iloc[folds[fold_order][1]]
df_validacao = data.iloc[folds[fold_order][2]]




# Separando os dados de entrada e saída.
X_treino = df_treino.drop(columns=['Churn'])
y_treino = df_treino['Churn']

X_test = df_test.drop(columns=['Churn'])
y_test = df_test['Churn']

X_val = df_validacao.drop(columns=['Churn'])
y_val = df_validacao['Churn']


scaler = StandardScaler()
X_train = scaler.fit_transform(X_treino)
y_train = y_treino

X_val = scaler.transform(X_val)
y_val = y_val

X_test = scaler.transform(X_test)
y_test = y_test

input_dim = X_train.shape[1]

def evaluate_model(model, X, y):
    # Predição das probabilidades
    y_pred_proba = model.predict(X, verbose=0)
    # Converter para 1D: shape (n_amostras,)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    # Converter y para numpy array 1D, se necessário
    y_true = y.to_numpy().ravel() if hasattr(y, "to_numpy") else np.array(y).ravel()

    # Cálculo das métricas
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

def objective(trial):
    model = Sequential()
    
    # Sugestão de hiperparâmetros
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    
    # Primeira camada com definição do input_dim
    units = trial.suggest_int('units_0', 16, 128, step=16)
    model.add(Dense(units, input_dim=input_dim, activation='relu'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Camadas ocultas adicionais
    for i in range(1, n_layers):
        units = trial.suggest_int(f'units_{i}', 16, 128, step=16)
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Camada de saída para classificação binária
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilação do modelo
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    
    # Treinamento
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Avaliação final do modelo
    val_metrics = evaluate_model(model, X_val, y_val)
    
    # Armazenar métricas adicionais no trial
    for metric_name, metric_value in val_metrics.items():
        trial.set_user_attr(f'val_{metric_name}', float(metric_value))
    
    # Retornamos o menor valor de loss de validação como métrica principal (minimização)
    best_val_auc = max(history.history['val_auc'])
    return best_val_auc

# Criação e execução do estudo com barra de progresso simplificada
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, show_progress_bar=True)

# Exibindo os melhores hiperparâmetros e resultados
print("\nMelhores hiperparâmetros:")
print(study.best_params)

print("\nMelhores métricas de validação:")
print(f"Loss: {study.best_value:.4f}")
for key, value in study.best_trial.user_attrs.items():
    print(f"{key}: {value:.4f}")

trials_df = study.trials_dataframe()
trials_df['param_observado'] =  'Maximaze AUC'



# Nome do arquivo
file_name = 'Projeto Final/MLP/optuna_results_MLP.xlsx'

# Verificar se o arquivo já existe
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