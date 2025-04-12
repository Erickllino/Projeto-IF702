import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

def preparar_dados(csv_path, target_col):
    # Carrega o CSV
    df = pd.read_csv(csv_path)

    # Separa features e alvo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y = y.map({"Yes": 1, "No": 0})

    # Identifica colunas categóricas e numéricas
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Divide 75% treino e 25% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50, stratify=y)

    # Pipeline de pré-processamento
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore"), cat_cols)
    ])

    # Ajusta com treino, transforma todos
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train.values.astype(np.float32), y_test.values.astype(np.float32)

def compute_ks_tf(y_true, y_pred_prob):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_prob = tf.reshape(y_pred_prob, [-1])
    sorted_indices = tf.argsort(y_pred_prob, direction='DESCENDING')
    y_true_sorted = tf.gather(y_true, sorted_indices)
    total_positives = tf.reduce_sum(y_true)
    total_negatives = tf.cast(tf.size(y_true), tf.float32) - total_positives
    tpr = tf.cumsum(y_true_sorted) / (total_positives + 1e-7)
    fpr = tf.cumsum(1 - y_true_sorted) / (total_negatives + 1e-7)
    ks_stat = tf.reduce_max(tf.abs(tpr - fpr))
    return ks_stat

def buildTabKANet():
    model = None
    # ...
    return model

def saveTrialData(trials_df):
    trials_df['param_observado'] = 'Maximize KS'
    file_name = 'Projeto Final/Tab-KAN/optuna_results_TABKANET.csv'
    if Path(file_name).exists():
        # Carregar dados existentes
        existing_df = pd.read_csv(file_name)
        
        # Combinar com novos dados e remover possíveis duplicatas
        combined_df = pd.concat([existing_df, trials_df], ignore_index=True)
        
        # Remover duplicatas baseado no ID do trial (se aplicável)
        if 'trial_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['trial_id'], keep='last')
        else:
            combined_df = combined_df.drop_duplicates()
        
        # Salvar dados combinados
        combined_df.to_csv(file_name, index=False)
        print("\nDados novos foram adicionados ao arquivo existente!")
    else:
        # Salvar pela primeira vez
        trials_df.to_csv(file_name, index=False)
        print("\nArquivo criado com os resultados dos testes do Optuna!")

def objective_tabKANet(trial):
    print(f">>> Iniciando Trial #{trial.number}.")
        
    params = {
        # ! DICIONÁRIO COM OS PARAMETROS A SEREM VARIADOS PELO OPTUNA AQUI
    }

    # ! CRIE O MODELO NESSE PONTO
    model = buildTabKANet()

    val_accuracies = []
    val_ks_values = []
    val_auc_values = []
    val_precision_values = []
    val_recall_values = []
    val_f1_values = []
    val_losses = []

    fold = 0
    for train, val in kf.split(X_train, y_train):
        fold += 1
        print(f">>> Fold: #{fold}.")
        
        fX_train = X_train[train]
        fy_train = y_train[train]
        fx_val = X_train[val]
        fy_val = y_train[val]
        
        model.fit(
            fX_train, 
            fy_train, 
            batch_size=params["batch_size"], 
            epochs=params["epochs"], 
            verbose=2, 
            callbacks=[es],
            validation_data=(fx_val, fy_val),
            shuffle=True,
            use_multiprocessing=True
            )

        y_val_proba = model.predict(fx_val, verbose=0).ravel()
        y_val_pred = (y_val_proba > 0.5).astype("int32")

        fpr_val, tpr_val, _ = roc_curve(fy_val, y_val_proba)
        val_ks = max(np.abs(tpr_val - fpr_val))
        val_acc = accuracy_score(fy_val, y_val_pred) 
        val_auc = roc_auc_score(fy_val, y_val_proba)
        val_precision = precision_score(fy_val, y_val_pred, zero_division=0)          
        val_recall = recall_score(fy_val, y_val_pred, zero_division=0)         
        val_f1 = f1_score(fy_val, y_val_pred, zero_division=0) 
        val_loss = model.evaluate(fx_val, fy_val, verbose=0)[0]

        val_accuracies.append(val_acc)
        val_ks_values.append(val_ks)         
        val_auc_values.append(val_auc)      
        val_precision_values.append(val_precision)        
        val_recall_values.append(val_recall)        
        val_f1_values.append(val_f1)
        val_losses.append(val_loss)

    mean_val_acc = np.mean(val_accuracies)
    mean_val_ks = np.mean(val_ks_values)
    mean_val_precision = np.mean(val_precision_values)
    mean_val_recall = np.mean(val_recall_values)
    mean_val_f1 = np.mean(val_f1_values)
    mean_val_auc = np.mean(val_auc_values)
    mean_val_loss = np.mean(val_losses)

    val_metrics = {
        'accuracy': mean_val_acc,
        'ks': mean_val_ks,
        'precision': mean_val_precision,
        'recall': mean_val_recall,
        'f1_score': mean_val_f1,
        'auc_roc': mean_val_auc,
        'loss': mean_val_loss
    }

    for metric_name, metric_value in val_metrics.items():
          trial.set_user_attr(f'val_{metric_name}', float(metric_value))

    print(f">>> Terminando Trial #{trial.number}.")

    if trial.number % 30 == 0:
        saveTrialData(trial.study.trials_dataframe())
        print(f">>> Processo parcial salvo, Trial #{trial.number}.")
    return mean_val_ks

ks_metric = tf.keras.metrics.MeanMetricWrapper(compute_ks_tf, name="ks")
kf = SKFold(n_splits=5, shuffle=True, random_state=50)
X_train, X_test, y_train, y_test = preparar_dados("Projeto Final/data/customer_churn_telecom_services.csv", "Churn")
es = tf.keras.callbacks.EarlyStopping(monitor="val_ks", patience=10, restore_best_weights=True, mode="max", verbose=1)

study = optuna.create_study(
    direction="maximize", 
    study_name="tabkanet2",
    storage="sqlite:///Projeto Final/Tab-KAN/tabkanet_trials.db",
    load_if_exists=True
)
study.optimize(objective_tabKANet, n_trials=None, show_progress_bar=False)
saveTrialData(study.trials_dataframe())
