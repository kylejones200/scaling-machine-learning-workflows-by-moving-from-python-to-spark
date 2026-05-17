"""Generated from Jupyter notebook: Spark example

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def notebook_step_001() -> None:
    spark = SparkSession.builder.appName("ML Pipeline Conversion").getOrCreate()


def notebook_step_002() -> None:
    spark


def time_jupyter_only() -> None:
    df = spark.read.csv(
        "/content/north_dakota_production.csv", header=True, inferSchema=True
    )

    df.printSchema()

    df.show(5)


def timeit_jupyter_only() -> None:
    pandas_df = pd.read_csv("/content/north_dakota_production.csv")

    spark_df = spark.createDataFrame(pandas_df)


def time_jupyter_only_2() -> None:
    pandas_df = pd.read_csv("/content/north_dakota_production.csv")


def time_jupyter_only_3() -> None:
    spark_df = spark.createDataFrame(pandas_df)


def notebook_step_007() -> None:
    spark_df.show(5)


def notebook_step_008() -> None:
    df.head(5)


def notebook_step_009() -> None:
    df.tail()


def notebook_step_010() -> None:
    pandas_df.head()


def time_jupyter_only_4() -> None:
    df = pd.read_csv("north_dakota_production.csv").dropna()

    df["HighProduction"] = (df["Oil"] > df["Oil"].median()).astype(int)

    features = ["Gas", "Wtr", "Days"]

    X = df[features]

    y = df["HighProduction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    print("Scikit-learn AUC:", auc)


def time_jupyter_only_5() -> None:
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("north_dakota_production.csv", header=True, inferSchema=True)

    median_oil = df.approxQuantile("Oil", [0.5], 0.01)[0]

    df = df.withColumn("HighProduction", when(df["Oil"] > median_oil, 1).otherwise(0))

    feature_cols = ["Gas", "Wtr", "Days"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="HighProduction")

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(labelCol="HighProduction")

    auc = evaluator.evaluate(predictions)

    print("Spark AUC:", auc)


def time_jupyter_only_6() -> None:
    df = pd.read_csv("north_dakota_production.csv")

    df = df[["Oil", "Gas", "Wtr", "Days"]].dropna()

    df["HighProduction"] = (df["Oil"] > df["Oil"].median()).astype(int)

    X = df[["Gas", "Wtr", "Days"]]

    y = df["HighProduction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    print("Scikit-learn AUC:", auc)


def time_jupyter_only_7() -> None:
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("north_dakota_production.csv", header=True, inferSchema=True)

    df = df.select("Oil", "Gas", "Wtr", "Days").na.drop()

    median_oil = df.approxQuantile("Oil", [0.5], 0.01)[0]

    df = df.withColumn("HighProduction", when(df["Oil"] > median_oil, 1).otherwise(0))

    assembler = VectorAssembler(inputCols=["Gas", "Wtr", "Days"], outputCol="features")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="HighProduction")

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(labelCol="HighProduction")

    auc = evaluator.evaluate(predictions)

    print("Spark AUC:", auc)


def notebook_step_015() -> None:
    len(df)


def notebook_step_016() -> None:
    predictions = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    auc = evaluator.evaluate(predictions)


def main() -> None:
    notebook_step_001()
    notebook_step_002()
    time_jupyter_only()
    timeit_jupyter_only()
    time_jupyter_only_2()
    time_jupyter_only_3()
    notebook_step_007()
    notebook_step_008()
    notebook_step_009()
    notebook_step_010()
    time_jupyter_only_4()
    time_jupyter_only_5()
    time_jupyter_only_6()
    time_jupyter_only_7()
    notebook_step_015()
    notebook_step_016()


if __name__ == "__main__":
    main()
