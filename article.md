---
author: "Kyle Jones"
date_published: "May 14, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/scaling-machine-learning-workflows-by-moving-from-python-to-spark-e5294640376a"
---

# Scaling Machine Learning Workflows by Moving from Python to Spark How to convert your scikit-learn pipelines to PySpark for faster,
distributed ML

### Scaling Machine Learning Workflows by Moving from Python to Spark
#### How to convert your scikit-learn pipelines to PySpark for faster, distributed ML
Python is slow. For data scientists or business analysts doing one experiment at a time, that slowness doesn't matter. And it is easy to trade some time for the benefit of easy use and plentiful analytics libraries like scikit-learn and pandas.

The problems with Python start to show when you try to scale AI/ML.

Apache Spark is a distributed computing framework designed for distributed data processing and ML. It allows you to write code once and run it anywhere --- on your laptop, a cloud cluster, or inside a managed platform like Databricks. Spark's MLlib package implements many of the common algorithms in sci-kit learn, but does so in a way designed for horizontal scalability (ie using more machines).

Cool idea. But if your code is all in Python, the fact that Spark is faster doesn't really help you.

My goal in this article is to show how you can easily migrate ML pipelines written in Python to Spark using PySpark.

#### Apache Spark MLlib
Apache Spark is a fault-tolerante, memory aware distributed data processing engine. Spark processes data using a directed acyclic graph (DAG) of tasks which optimizes execution plans for performance.

Spark MLlib is the ML library that builds on Spark's DataFrame API and can perform things like linear regression, logistic regression, decision trees, and clustering. Scikit-learn is based on NumPy arrays. Spark MLlib uses Spark DataFrames which are a column of feature vectors. That change lets MLlib do automatic optimization, lazy evaluation, and parallel processing.

Spark MLlib focuses on immutability, pipeline composition, and scalability. The pipeline includes all preprocessing and modeling steps, then fits the entire pipeline at once. This approach closely mirrors scikit-learn's `Pipeline` API but is optimized for distributed data (meaning it is faster).

### Setting Up Spark with PySpark
To get started with PySpark, you'll need to install it and initialize a Spark session. On most machines:

``` 
pip install pyspark
```

You can then create a session:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ML Pipeline Conversion") \
    .getOrCreate()
```

Once initialized, Spark provides access to DataFrame APIs similar to pandas, but with distributed backend execution.

You can read data using:

``` 
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)
```

Always check the schema and confirm that numeric columns were interpreted correctly. String columns will need to be indexed for MLlib.

### Converting Python DataFrames to Spark
If you're starting with a pandas DataFrame, you can convert it:

```python
import pandas as pd
from pyspark.sql import SparkSession

pandas_df = pd.read_csv("data.csv")
spark_df = spark.createDataFrame(pandas_df)
```

Watch out for mixed data types. Spark prefers clearly typed columns: strings, integers, doubles, booleans. You need to deal with nulls explicitly:

``` 
spark_df = spark_df.na.fill(0)  # or use dropna()
```

For machine learning, all features must be numeric and assembled into a single vector column. This differs from scikit-learn, where the model can accept multiple numeric columns directly.

### Rewriting ML Pipelines in Spark
In scikit-learn, you might write:

```python
%%time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Load your DataFrame
df = pd.read_csv("north_dakota_production.csv")

# Drop NA rows
df = df[['Oil', 'Gas', 'Wtr', 'Days']].dropna()

# Create binary target
df['HighProduction'] = (df['Oil'] > df['Oil'].median()).astype(int)

# Define features and target
X = df[['Gas', 'Wtr', 'Days']]
y = df['HighProduction']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Evaluate
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("Scikit-learn AUC:", auc)
```

In Spark:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Spark session
spark = SparkSession.builder.getOrCreate()

# Load data
df = spark.read.csv("north_dakota_production.csv", header=True, inferSchema=True)

# Drop NA rows
df = df.select("Oil", "Gas", "Wtr", "Days").na.drop()

# Add binary target column
median_oil = df.approxQuantile("Oil", [0.5], 0.01)[0]
df = df.withColumn("HighProduction", when(df["Oil"] > median_oil, 1).otherwise(0))

# Features
assembler = VectorAssembler(inputCols=["Gas", "Wtr", "Days"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="HighProduction")

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Split and train
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="HighProduction")
auc = evaluator.evaluate(predictions)
print("Spark AUC:", auc)
```

### Advanced Scaling Patterns
For tuning:

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
cvModel = crossval.fit(train_data)
```

To use Spark Streaming with ML, you can ingest live data and apply a trained model:

``` 
stream = spark.readStream.format("csv").option("path", "live_data/").load()
predictions = cvModel.transform(stream)
```

### When Not to Use Spark for ML
Spark introduces complexity. Scikit-learn may be better if your data fits in memory or your model needs interpretability,. Custom algorithms, such as deep learning with PyTorch or complex ensemble models, are also better handled outside Spark.

Spark MLlib supports only a subset of commonly used models. If you need flexible architectures, transfer learning, or neural nets, Spark won't be ideal.

Here are the results from a binary classificication of 40,000 oil wells in North Dakota. The dataset has 1,932,995 and is 438MBs as a CSV file.


The dataset is too small to get the value of spark.

Migrating ML from Python to Spark helps when your data becomes too large to handle on a single machine. Spark MLlib supports many familiar algorithms and offers a clean way to express full pipelines at scale.

The tradeoff is setup complexity and the need to use Spark's API, which differs from the scikit-learn idioms you're used to. However, the performance gains for large data make it worthwhile.

Databricks, AWS Glue, and MLflow are better tools than Colab for production-ready pipelines with Spark.
