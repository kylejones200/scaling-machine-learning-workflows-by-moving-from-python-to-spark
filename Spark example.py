"""Generated from Jupyter notebook: Spark example

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("ML Pipeline Conversion").getOrCreate()
    spark
    df = spark.read.csv(
        "/content/north_dakota_production.csv", header=True, inferSchema=True
    )
    df.printSchema()
    df.show(5)
    import pandas as pd
    from pyspark.sql import SparkSession

    pandas_df = pd.read_csv("/content/north_dakota_production.csv")
    spark_df = spark.createDataFrame(pandas_df)
    pandas_df = pd.read_csv("/content/north_dakota_production.csv")
    spark_df = spark.createDataFrame(pandas_df)
    spark_df.show(5)
    df.head(5)
    df.tail()
    pandas_df.head()
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import when

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
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import when

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
    len(df)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)


def main() -> None:
    main()


if __name__ == "__main__":
    main()
