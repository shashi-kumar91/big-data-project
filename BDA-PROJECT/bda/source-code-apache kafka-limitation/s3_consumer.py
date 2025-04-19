from kafka import KafkaConsumer
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Kafka Configuration
KAFKA_TOPIC = "y_topic"  # Ensure this matches the producer
KAFKA_BROKER = "localhost:9092"

# Model Path
model_path = "/home/iiitdmk-aidslab-23/Downloads/singlemodel(3).h5"

# ✅ Load Model
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file '{model_path}' not found.")
    exit(1)

try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
    print(f"🔍 Model expected input shape: {model.input_shape}")
    expected_features = model.input_shape[2]  # Shape: (None, 1, 989)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# ✅ Initialize a New Standard Scaler
scaler = StandardScaler()
print("✅ Initialized new StandardScaler for 989 features.")

# ✅ Kafka Consumer Setup
try:
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest"
    )
    print(f"🚀 Kafka Consumer connected, listening on topic: {KAFKA_TOPIC}")
except Exception as e:
    print(f"❌ ERROR connecting to Kafka: {e}")
    exit(1)

# ✅ Process Incoming Messages
scaler_fitted = False  # Flag to track if scaler is fitted
for msg in consumer:
    try:
        data = msg.value
        print(f"📝 Raw message received: {data}")  # Debug raw message
        received_features = len(data["features"])
        print(f"📏 Received {received_features} features, expected {expected_features}")

        if received_features != expected_features:
            print(f"⚠️ WARNING: Received {received_features} features, expected {expected_features}. Skipping.")
            continue

        # ✅ Preprocess Input
        X_input = np.array(data["features"]).reshape(1, expected_features)
        X_input = np.expand_dims(X_input, axis=1)  # Shape: (1, 1, 989)

        # Fit scaler incrementally on first message (or small batch)
        if not scaler_fitted:
            scaler.partial_fit(X_input.reshape(1, -1))  # Fit on first message
            scaler_fitted = True
            print("✅ Scaler fitted with first message.")

        # Apply Scaling
        X_input = scaler.transform(X_input.reshape(1, -1))  # Flatten for scaler
        X_input = X_input.reshape((1, 1, expected_features))  # Reshape back

        # 🔮 Make Prediction
        prediction = model.predict(X_input, verbose=0)[0][0]
        predicted_label = int(prediction >= 0.5)

        # ✅ Print Results
        print(f"✅ Received Data: Features={len(data['features'])}, Label={data['label']}")
        print(f"🎯 Predicted Label: {predicted_label}, Confidence: {prediction:.4f}\n")

    except Exception as e:
        print(f"❌ ERROR processing message: {e}")
        continue

# Cleanup
consumer.close()
print("✅ Kafka Consumer closed.")
