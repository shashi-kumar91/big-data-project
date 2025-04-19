import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import udf, when
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, DoubleType
from pyspark.ml.classification import LogisticRegression
import pickle
import time

# Initialize Spark Session with HDFS config
spark = SparkSession.builder \
    .appName("SBOA Feature Selection") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "6") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://master:9000") \
    .getOrCreate()

# Load Epsilon dataset
start_time = time.time()
dataset_path = "hdfs://master:9000/user/ubuntu1/mydir/epsilon_normalized.t"
df = spark.read.format("libsvm").load(dataset_path).cache()
print(f"Loading dataset took {time.time() - start_time:.2f} seconds")

# Convert labels from -1/1 to 0/1
start_time = time.time()
df = df.withColumn("label", when(df["label"] == -1, 0).otherwise(1))
print(f"Label conversion took {time.time() - start_time:.2f} seconds")

# Sample data for SBOA
start_time = time.time()
sample_df = df.sample(False, 0.005, seed=42)
sample_data = sample_df.rdd.map(lambda row: (np.array(row["features"].toArray()), row["label"])).collect()
X_sample = np.array([x[0] for x in sample_data])
y_sample = np.array([x[1] for x in sample_data])
print(f"Sampling took {time.time() - start_time:.2f} seconds, Sample size: {len(X_sample)}")

# SBOA Algorithm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate_fitness(features_subset, X, y, lr_model=None):
    selected_indices = np.where(features_subset)[0]
    if len(selected_indices) == 0:
        return float('inf')
    X_selected = X[:, selected_indices]
    df_train = spark.createDataFrame(
        [(DenseVector(x), float(y)) for x, y in zip(X_selected, y)],
        ["features", "label"]
    ).repartition(50)
    if lr_model is None:
        lr = LogisticRegression(maxIter=5)
        model = lr.fit(df_train)
    else:
        model = lr_model.setFeaturesCol("features").fit(df_train)
    predictions = model.transform(df_train)
    accuracy = predictions.filter(predictions["prediction"] == predictions["label"]).count() / predictions.count()
    error_rate = 1 - accuracy
    num_features = len(selected_indices)
    total_features = 2000
    alpha, beta = 0.8, 0.2
    return alpha * error_rate + beta * (num_features / total_features)

def sboa_feature_selection(X, y, num_iterations=10, num_butterflies=3):
    num_features = X.shape[1]
    butterflies = np.random.randint(0, 2, size=(num_butterflies, num_features))
    best_solution = butterflies[0].copy()
    start_time = time.time()
    lr = LogisticRegression(maxIter=5)
    best_fitness = evaluate_fitness(best_solution, X, y, lr)
    print(f"Initial fitness evaluation took {time.time() - start_time:.2f} seconds")

    for t in range(num_iterations):
        iter_start = time.time()
        for i in range(num_butterflies):
            fitness = evaluate_fitness(butterflies[i], X, y, lr)
            fragrance = sigmoid(fitness)
            r = np.random.random()
            if r < fragrance:
                butterflies[i] = np.random.randint(0, 2, num_features)
            else:
                j = np.random.randint(0, num_features)
                butterflies[i][j] = 1 - butterflies[i][j]
            current_fitness = evaluate_fitness(butterflies[i], X, y, lr)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = butterflies[i].copy()
        print(f"Iteration {t+1}/{num_iterations} took {time.time() - iter_start:.2f} seconds")
    return best_solution

# Apply SBOA
start_time = time.time()
selected_features = sboa_feature_selection(X_sample, y_sample)
selected_feature_indices = np.where(selected_features)[0]
print(f"SBOA took {time.time() - start_time:.2f} seconds")
print(f"Selected {len(selected_feature_indices)} features out of 2000")

# Filter full dataset
start_time = time.time()
selected_features_broadcast = spark.sparkContext.broadcast(selected_feature_indices)
def filter_features(features, label):
    filtered = np.array(features.toArray())[selected_features_broadcast.value].tolist()
    return (filtered, float(label))

filtered_rdd = df.rdd.map(lambda row: filter_features(row["features"], row["label"]))
schema = StructType([
    StructField("features", ArrayType(FloatType()), True),
    StructField("label", DoubleType(), True)
])

df_selected = spark.createDataFrame(filtered_rdd, schema).repartition(50)
print(f"Filtering took {time.time() - start_time:.2f} seconds")

# Write output
start_time = time.time()
df_selected.write.parquet("hdfs://master:9000/user/ubuntu1/mydir/epsilon_selected.parquet", mode="overwrite")
with open("selected_features.pkl", "wb") as f:
    pickle.dump(selected_feature_indices, f)
print(f"Writing output took {time.time() - start_time:.2f} seconds")

spark.stop()