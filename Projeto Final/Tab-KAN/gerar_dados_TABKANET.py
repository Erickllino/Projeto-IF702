
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import optuna
from pathlib import Path
from sklearn.metrics import roc_curve
import openpyxl

numTrial = 0
df_source = pd.read_csv('Projeto Final/data/customer_churn_telecom_services.csv')
kf = SKFold(n_splits=5, shuffle=True, random_state=50)
training_history = []

def compute_ks(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    return max(abs(tpr - fpr))

def load_and_preprocess_data(df: pd.DataFrame, target_col="Churn", test_size=0.25, random_state=42):
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    # Separar features e alvo
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Aplicar One-Hot Encoding para variáveis categóricas
    X_encoded = pd.get_dummies(X)
    
    # Escalonar as features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

class KSEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, patience=10, min_delta=1e-4):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.min_delta = min_delta
        self.best_ks = -float('inf')
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0).ravel()
        ks = compute_ks(self.y_val, y_pred)
        print(f"Epoch {epoch + 1}: KS={ks:.4f}")

        if ks > self.best_ks + self.min_delta:
            self.best_ks = ks
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered. Best KS={self.best_ks:.4f}")
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
        
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-8))  # evitar divisão por zero

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def build_tabkanet_model(params, input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = inputs
    for i in range(params["num_layers"]):
        x = tf.keras.layers.Dense(params["units"], params["activation"])(x)
        if params["dropout_rate"] > 0:
            x = tf.keras.layers.Dropout(params["dropout_rate"])(x)
    outputs = tf.keras.layers.Dense(1, params["activation"])(x)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=[
                        tf.keras.metrics.MeanSquaredError(),
                        tf.keras.metrics.AUC(name="auc"),
                        tf.keras.metrics.Recall(name="recall"),
                        tf.keras.metrics.Precision(name="precision"),
                        F1Score(name="f1_score")
                        ])
    return model

def optimize_hyperparameters(X_train, y_train, n_trials):
    input_dim = X_train.shape[1]
    activation_map = {
        "relu": tf.keras.activations.relu,
        "sigmoid": tf.keras.activations.sigmoid,
        "tanh": tf.keras.activations.tanh,
        "softmax": tf.keras.activations.softmax
    }
    # es = tf.keras.callbacks.EarlyStopping(monitor="val_ks_stat", patience=10, restore_best_weights=True, mode="max")
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train )
    
    def objective(trial):
        global numTrial
        numTrial += 1
        print(f">>> Iniciando Trial {numTrial}.")
        params = {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "units": trial.suggest_int("units", 16, 128, step=16),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"]),
            "epochs": trial.suggest_int("epochs", 10, 100, step=10),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

        params["activation"] = activation_map[params["activation"]]
        ks_callback = KSEarlyStopping((X_val, y_val), patience=10)
        model = build_tabkanet_model(params, input_dim)

        print(f">>> Iniciando Treino {numTrial}.")
        model.fit(
            X_train_inner, y_train_inner,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=[ks_callback]
        )
        print(f">>> Terminando Treino {numTrial}.")

        y_pred_prob = model.predict(X_val, verbose=0).ravel()
        ks_val = compute_ks(y_val, y_pred_prob)

        print(f">>> Terminando Trial {numTrial}.")
        return ks_val
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

X_train_tabkan, X_test_tabkan, y_train_tabkan, y_test_tabkan = load_and_preprocess_data(df_source)
study = optimize_hyperparameters(X_train_tabkan, y_train_tabkan, 1)
trials_df = study.trials_dataframe()
trials_df['param_observado'] = 'Maximize KS'

file_name = 'Projeto Final/Tab-KAN/optuna_results_TABKANET.xlsx'

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
