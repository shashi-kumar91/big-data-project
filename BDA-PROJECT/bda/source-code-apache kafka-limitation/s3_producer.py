from kafka import KafkaProducer
import json
import pyarrow.parquet as pq
import numpy as np
import time
import os

# Kafka Configuration
KAFKA_TOPIC = "y_topic"  # Change to "y_topic_v2" if creating a new topic
KAFKA_BROKER = "localhost:9092"

# Load Preprocessed Data
parquet_path = "/home/iiitdmk-aidslab-23/Downloads/epsilon_selected.parquet/part-00000-ccea8c8d-4f96-4c1c-9a16-e19f367bc425-c000.snappy.parquet"

# Check if the Parquet file exists
if not os.path.exists(parquet_path):
    print(f"❌ ERROR: Parquet file '{parquet_path}' not found.")
    exit(1)

# Kafka Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        buffer_memory=33554432,
        batch_size=16384,
        linger_ms=5
    )
    print("✅ Kafka Producer connected.")
except Exception as e:
    print(f"❌ ERROR connecting to Kafka: {e}")
    exit(1)

# Stream Data in Batches
BATCH_SIZE = 1000
EXPECTED_FEATURES = 989
print(f"🚀 Sending data to Kafka topic: {KAFKA_TOPIC}")

try:
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    print(f"📊 Total rows in Parquet file: {num_rows}")

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        print(f"📑 Columns in dataframe: {df.columns}")
        print(f"🔍 Null values in data: \n{df.isnull().sum()}")

        try:
            X_batch = np.stack(df["features"].values)[:, :EXPECTED_FEATURES]
            y_batch = df["label"].values.astype(np.float32)
        except KeyError as e:
            print(f"❌ ERROR: Missing expected columns in Parquet file: {e}")
            exit(1)

        print(f"📐 X_batch shape: {X_batch.shape}")
        print(f"📐 y_batch shape: {y_batch.shape}")

        for i in range(len(X_batch)):
            message = {"features": X_batch[i].tolist(), "label": int(y_batch[i])}
            print(f"📝 Sending message with {len(message['features'])} features")  # Debug feature count
            try:
                producer.send(KAFKA_TOPIC, value=message)
                print(f"📤 Sent message {i+1}/{len(X_batch)} in batch.")
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ ERROR sending message: {e}")

        producer.flush()
        print(f"✅ Flushed batch of {len(X_batch)} messages.")

except Exception as e:
    print(f"❌ ERROR processing Parquet file: {e}")
    exit(1)

producer.flush()
producer.close()
print("✅ Kafka Producer finished sending messages.")
