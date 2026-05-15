"""Generated from Jupyter notebook: Spark example

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("ML Pipeline Conversion").getOrCreate()


    # --- code cell ---

    spark


    # --- code cell ---

    # %time  # Jupyter-only
    df = spark.read.csv(
        "/content/north_dakota_production.csv", header=True, inferSchema=True
    )
    df.printSchema()
    df.show(5)


    # --- code cell ---

    # %timeit  # Jupyter-only
    import pandas as pd
    from pyspark.sql import SparkSession

    pandas_df = pd.read_csv("/content/north_dakota_production.csv")
    spark_df = spark.createDataFrame(pandas_df)


    # --- code cell ---

    # %time  # Jupyter-only

    pandas_df = pd.read_csv("/content/north_dakota_production.csv")


    # --- code cell ---

    # %time  # Jupyter-only
    spark_df = spark.createDataFrame(pandas_df)


    # --- code cell ---

    spark_df.show(5)


    # --- code cell ---

    df.head(5)


    # --- code cell ---

    df.tail()


    # --- code cell ---

    pandas_df.head()


    # --- code cell ---

    # %time  # Jupyter-only
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Load your DataFrame
    df = pd.read_csv("north_dakota_production.csv").dropna()  # Assume you converted to CSV

    # Define target: high vs low production
    df["HighProduction"] = (df["Oil"] > df["Oil"].median()).astype(int)

    # Select numerical features
    features = ["Gas", "Wtr", "Days"]
    X = df[features]
    y = df["HighProduction"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Scikit-learn pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Scikit-learn AUC:", auc)


    # --- code cell ---

    # %time  # Jupyter-only

    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import when

    # Spark session
    spark = SparkSession.builder.getOrCreate()

    # Load DataFrame (replace with your actual path or DataFrame)
    df = spark.read.csv("north_dakota_production.csv", header=True, inferSchema=True)

    # Add target column: HighProduction
    median_oil = df.approxQuantile("Oil", [0.5], 0.01)[0]
    df = df.withColumn("HighProduction", when(df["Oil"] > median_oil, 1).otherwise(0))

    # Select relevant features
    feature_cols = ["Gas", "Wtr", "Days"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    # Logistic regression
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="HighProduction")

    # Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Split
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Fit model
    model = pipeline.fit(train_data)

    # Predict
    predictions = model.transform(test_data)

    # Evaluate
    evaluator = BinaryClassificationEvaluator(labelCol="HighProduction")
    auc = evaluator.evaluate(predictions)
    print("Spark AUC:", auc)


    # --- code cell ---

    # %%time  # Jupyter-only
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Load your DataFrame
    df = pd.read_csv("north_dakota_production.csv")

    # Drop NA rows
    df = df[["Oil", "Gas", "Wtr", "Days"]].dropna()

    # Create binary target
    df["HighProduction"] = (df["Oil"] > df["Oil"].median()).astype(int)

    # Define features and target
    X = df[["Gas", "Wtr", "Days"]]
    y = df["HighProduction"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Scikit-learn AUC:", auc)


    # --- code cell ---

    # %%time  # Jupyter-only
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import when

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


    # --- code cell ---

    len(df)


    # --- code cell ---

    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)


if __name__ == "__main__":
    main()
