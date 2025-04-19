import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pyarrow.parquet as pq
import time
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load preprocessed data
start_time = time.time()
parquet_path = "D:\\bda_project\\epsilon_selected.parquet"
table = pq.read_table(parquet_path)
df = table.to_pandas()
X_all = np.stack(df["features"].values)
y_all = df["label"].values.astype(np.float32)
print(f"Loading parquet took {time.time() - start_time:.2f} seconds")

# Split data (80% train, 20% test)
train_size = 400000 if len(X_all) >= 500000 else int(len(X_all) * 0.8)
test_size = 100000 if len(X_all) >= 500000 else len(X_all) - train_size
indices = np.random.permutation(len(X_all))
X_train, X_test = X_all[indices[:train_size]], X_all[indices[train_size:train_size + test_size]]
y_train, y_test = y_all[indices[:train_size]], y_all[indices[train_size:train_size + test_size]]
print(f"Data split: {len(X_train)} train, {len(X_test)} test, took {time.time() - start_time:.2f} seconds")

# Reshape for GRU
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Optimized OGRU Model
def build_optimized_ogru(input_shape):
    model = Sequential([
        GRU(units=128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        GRU(units=64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
start_time = time.time()
model = build_optimized_ogru((1, X_train.shape[2]))
history = model.fit(
    X_train, y_train, epochs=50, batch_size=512,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)
print(f"Training took {time.time() - start_time:.2f} seconds")

# Evaluate
start_time = time.time()
loss, accuracy, auc = model.evaluate(X_test, y_test)
print(
    f"Test Accuracy: {accuracy * 100:.2f}%, Test AUC: {auc * 100:.2f}%, Evaluation took {time.time() - start_time:.2f} seconds")

# Compute AUC explicitly
y_pred_proba = model.predict(X_test).ravel()
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Computed AUC Score: {auc_score * 100:.2f}%")

# Save the model
model.save("optimized_ogru_model.h5")
