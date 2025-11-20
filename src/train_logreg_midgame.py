#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from chess_spark_pipeline import FEATURE_KEYS


def create_spark_session(app_name: str = "chess-logreg-midgame") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


if __name__ == "__main__":
    # Ruta del parquet generado por main_local.py
    INPUT_PATH = "../dataset/Lichess_2013_2014_features_50.parquet"

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Cargar dataset de features
    df = spark.read.parquet(INPUT_PATH)

    # 2) Definir columnas de entrada (solo estructura/posición + total_moves)
    #    Usamos las mismas FEATURE_KEYS de chess_spark_pipeline + total_moves
    feature_cols = FEATURE_KEYS + ["total_moves"]

    # 3) Construir dataset de modelado: label + features numéricas
    df_model = (
        df.select(
            ["label_white_win"] + feature_cols
        )
        .dropna()  # quitar filas con algún None/NaN
        .filter(F.col("label_white_win").isin(0, 1))
    )

    print(f"Numero de filas tras limpieza: {df_model.count()}")

    # 4) VectorAssembler + StandardScaler + LogisticRegression
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_white_win",
        maxIter=100,
        regParam=0.1,       # λ de regularización L2
        elasticNetParam=0.0 # 0 = L2 pura
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # 5) Train / test split
    train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

    print(f"Train: {train_df.count()} filas, Test: {test_df.count()} filas")

    # 6) Entrenar
    model = pipeline.fit(train_df)

    # 7) Evaluar en test
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label_white_win",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    auc = evaluator.evaluate(predictions)
    print(f"\nAUC ROC (test): {auc:.4f}")

    # También accuracy como referencia
    accuracy = (
        predictions
        .withColumn("correct", F.when(
            F.col("prediction") == F.col("label_white_win"), 1.0
        ).otherwise(0.0))
        .agg(F.avg("correct"))
        .first()[0]
    )
    print(f"Accuracy (test): {accuracy:.4f}")

    # 8) Interpretar coeficientes
    lr_model = model.stages[-1]  # última etapa del pipeline

    coeffs = lr_model.coefficients.toArray()
    intercept = lr_model.intercept

    print(f"\nIntercepto (b): {intercept:.4f}\n")

    # Emparejar features con coeficientes y ordenarlos por impacto |w|
    coef_with_names = list(zip(feature_cols, coeffs))
    coef_sorted = sorted(coef_with_names, key=lambda x: abs(x[1]), reverse=True)

    print("Top 20 features por |coeficiente|:")
    for name, w in coef_sorted[:20]:
        print(f"{name:30s} -> w = {w:+.4f}")

    spark.stop()
