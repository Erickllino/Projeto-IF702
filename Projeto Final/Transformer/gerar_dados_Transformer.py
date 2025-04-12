import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.exceptions import TrialPruned
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Imports do TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, metrics
# Callback de Pruning do Optuna para Keras
from optuna.integration import KerasPruningCallback




df = pd.read_csv('Projeto Final/data/customer_churn_telecom_services.csv')


df = df.drop_duplicates(ignore_index=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.fillna(0)

df = pd.get_dummies(df, drop_first=True, dummy_na=False)


target_column = 'Churn_Yes'
if target_column not in df.columns:
    print(f"Erro: Coluna target '{target_column}' não encontrada no DataFrame após get_dummies.")
    print("Colunas disponíveis:", df.columns.tolist())
    exit()

df_data = df.drop(columns=target_column).astype(np.float32)
df_target = df[target_column].astype(np.int32)

# Validação cruzada
N_SPLITS = 5
kf = SKFold(n_splits=N_SPLITS, shuffle=True, random_state=50)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Atenção e normalização
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # Conexão residual

    # Feed-forward e normalização
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Utiliza a dimensão estática para definir o número de filtros
    out_dim = tf.keras.backend.int_shape(inputs)[-1]
    x = layers.Conv1D(filters=out_dim, kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, learning_rate=1e-3):
    """Cria um modelo Keras com blocos Transformer para classificação."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Opcional: camada densa inicial para projeção das features
    # x = layers.Dense(head_size * num_heads)(x)

    # Cria blocos Transformer
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Pooling para reduzir a dimensionalidade antes do classificador
    x = layers.GlobalAveragePooling1D()(x)  # Usa o padrão 'channels_last'

    # Camadas MLP (classificador) no final
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # Camada de saída para classificação binária
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.AUC(name='auc'), metrics.BinaryAccuracy(name='accuracy'),
                 metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    )
    return model



def objective_transformer(trial):
    # Hiperparâmetros a serem otimizados
    head_size = trial.suggest_categorical("head_size", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 1, 4)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)

    mlp_units = []
    for i in range(num_hidden_layers):
        units = trial.suggest_int(f"units_layer_{i}", 32, 256, step=32)
        mlp_units.append(units)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 100  # Máximo de épocas

    # Callback de EarlyStopping
    early_stopping = callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)

    # Listas para armazenar métricas por fold
    val_accuracies = []
    val_ks_values = []
    val_auc_values = []
    val_precision_values = []
    val_recall_values = []
    val_f1_values = []

    for fold, (train_index, val_index) in enumerate(kf.split(df_data, df_target)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_val = df_data.iloc[train_index], df_data.iloc[val_index]
        y_train, y_val = df_target.iloc[train_index], df_target.iloc[val_index]

        # Escalonamento dos dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Ajuste de dimensão para Keras (adiciona dimensão de 'sequência' de tamanho 1)
        X_train_reshaped = np.expand_dims(X_train_scaled, axis=1)
        X_val_reshaped = np.expand_dims(X_val_scaled, axis=1)

        input_shape = X_train_reshaped.shape[1:]

        # Limpa a sessão do Keras para evitar vazamento de memória
        keras.backend.clear_session()

        # Constrói o modelo
        model = build_transformer_model(
            input_shape=input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            learning_rate=learning_rate
        )

        # Treinamento do modelo
        history = model.fit(
            X_train_reshaped, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_reshaped, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Avaliação
        y_val_proba = model.predict(X_val_reshaped, verbose=0).flatten()
        y_val_pred = (y_val_proba > 0.5).astype(int)

        val_acc = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        # Cálculo do KS (diferença máxima entre TPR e FPR)
        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
        val_ks = max(np.abs(tpr_val - fpr_val)) if len(tpr_val) > 0 and len(fpr_val) > 0 else 0.0

        # Armazenamento dos resultados
        val_accuracies.append(val_acc)
        val_ks_values.append(val_ks)
        val_auc_values.append(val_auc)
        val_precision_values.append(val_precision)
        val_recall_values.append(val_recall)
        val_f1_values.append(val_f1)

    # Métricas médias dos folds
    mean_val_acc = np.mean(val_accuracies)
    mean_val_ks = np.mean(val_ks_values)
    mean_val_auc = np.mean(val_auc_values)
    mean_val_precision = np.mean(val_precision_values)
    mean_val_recall = np.mean(val_recall_values)
    mean_val_f1 = np.mean(val_f1_values)

    # Guarda as métricas como atributos do trial
    trial.set_user_attr('val_accuracy', mean_val_acc)
    trial.set_user_attr('val_ks', mean_val_ks)
    trial.set_user_attr('val_auc', mean_val_auc)
    trial.set_user_attr('val_precision', mean_val_precision)
    trial.set_user_attr('val_recall', mean_val_recall)
    trial.set_user_attr('val_f1', mean_val_f1)

    return mean_val_ks  # Retorne a métrica que deseja otimizar

# Cria o estudo com a direção de maximização
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())

# Otimização
N_TRIALS = 500  # Ajuste conforme necessário
study.optimize(objective_transformer, n_trials=N_TRIALS, show_progress_bar=True)

print("\nOtimização concluída!")
print("Melhor Trial:")
best_trial = study.best_trial
print(f"  Valor (AUC Médio): {best_trial.value:.4f}")
print("  Parâmetros:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("  Métricas Médias de Validação:")
for key, value in best_trial.user_attrs.items():
    print(f"    {key}: {value:.4f}")

trials_df = study.trials_dataframe()
trials_df['param_observado'] = 'Maximize AUC'  # ou 'Maximize KS'

# Renomear colunas para clareza (opcional)
trials_df = trials_df.rename(columns={
    'value': 'objective_value (mean_val_auc)',
    'user_attrs_val_accuracy': 'mean_val_accuracy',
    'user_attrs_val_ks': 'mean_val_ks',
    'user_attrs_val_auc': 'mean_val_auc_attr',
    'user_attrs_val_precision': 'mean_val_precision',
    'user_attrs_val_recall': 'mean_val_recall',
    'user_attrs_val_f1': 'mean_val_f1'
})
trials_df = trials_df.drop(columns=['datetime_start', 'datetime_complete', 'duration', 'state'], errors='ignore')

# Cria pasta de saída se necessário e salva os resultados em Excel
output_dir = Path('Projeto Final/Transformer')
output_dir.mkdir(parents=True, exist_ok=True)
file_name = output_dir / 'optuna_results_Transformer.xlsx'

try:
    if file_name.exists():
        print(f"\nArquivo '{file_name}' existe. Adicionando novos trials...")
        existing_df = pd.read_excel(file_name)
        for col in trials_df.columns:
            if trials_df[col].dtype == 'object':
                trials_df[col] = trials_df[col].astype(str)
        for col in existing_df.columns:
            if existing_df[col].dtype == 'object':
                existing_df[col] = existing_df[col].astype(str)
        combined_df = pd.concat([existing_df, trials_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['number'], keep='last')
        combined_df.to_excel(file_name, index=False)
        print("Dados novos adicionados e duplicatas removidas.")
    else:
        trials_df.to_excel(file_name, index=False)
        print(f"\nArquivo '{file_name}' criado com os resultados!")
except Exception as e:
    print(f"\nErro ao salvar o arquivo Excel: {e}")
    print("Salvando resultados em um arquivo CSV como backup: results_backup.csv")
    trials_df.to_csv('results_backup.csv', index=False)
