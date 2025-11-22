#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import functions as F

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from chess_spark_pipeline import (
    create_spark_session,
    FEATURE_KEYS,
)


def train_logreg_for_phase(df, feature_cols, phase_name: str):
    """
    Entrena una regresión logística binaria para una fase concreta
    (apertura / midgame / final) y muestra métricas + coeficientes.
    """
    print(f"\n==================== {phase_name.upper()} ====================")

    # 1) Construir DataFrame con label + features
    cols = ["label_white_win"] + feature_cols
    df_phase = (
        df.select(cols)
        .dropna()
        .filter(F.col("label_white_win").isin(0, 1))
    )

    n_rows = df_phase.count()
    print(f"Filas válidas para {phase_name}: {n_rows}")

    if n_rows < 50:
        print("Advertencia: muy pocas filas, las métricas pueden ser inestables.")

    # Distribución de clases
    print("Distribución de clases (label_white_win):")
    df_phase.groupBy("label_white_win").count().show()

    # 2) Ensamblar y escalar features
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
        regParam=0.1,
        elasticNetParam=0.0,  # L2 pura
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # 3) Train / test split
    train_df, test_df = df_phase.randomSplit([0.8, 0.2], seed=42)

    print(f"Train: {train_df.count()} filas, Test: {test_df.count()} filas")

    if test_df.count() == 0 or train_df.count() == 0:
        print("Muy pocas filas para train/test, se omite entrenamiento.")
        return

    # 4) Entrenar
    model = pipeline.fit(train_df)

    # 5) Evaluar en test
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label_white_win",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    auc = evaluator.evaluate(predictions)
    print(f"AUC ROC (test) [{phase_name}]: {auc:.4f}")

    # Accuracy
    accuracy = (
        predictions
        .withColumn(
            "correct",
            F.when(
                F.col("prediction") == F.col("label_white_win"),
                1.0,
            ).otherwise(0.0),
        )
        .agg(F.avg("correct"))
        .first()[0]
    )
    print(f"Accuracy (test) [{phase_name}]: {accuracy:.4f}")

    # 6) Coeficientes
    lr_model = model.stages[-1]  # última etapa del pipeline

    coeffs = lr_model.coefficients.toArray()
    intercept = lr_model.intercept

    print(f"\nIntercepto (b) [{phase_name}]: {intercept:.4f}\n")

    coef_with_names = list(zip(feature_cols, coeffs))
    coef_sorted = sorted(coef_with_names, key=lambda x: abs(x[1]), reverse=True)

    print(f"Top 20 features por |coeficiente| para {phase_name}:")
    for name, w in coef_sorted[:20]:
        print(f"{name:30s} -> w = {w:+.4f}")


if __name__ == "__main__":
    # Ruta del parquet generado por main_local.py
    INPUT_PATH = "../experimentos/Lichess_2013_2014_features_full.parquet"

    spark: SparkSession = create_spark_session("chess-logreg-phases")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(INPUT_PATH)

    # ---------------- MIDGAME (fen_after_20_moves, sin prefijo) ----------------
    mid_features = FEATURE_KEYS  # columnas sin prefijo: material_diff, pawns_diff, ...

    train_logreg_for_phase(df, mid_features, phase_name="midgame (move 20)")

    # ---------------- APERTURA (prefijo open_) ----------------
    open_features = [f"open_{k}" for k in FEATURE_KEYS]

    train_logreg_for_phase(df, open_features, phase_name="opening")

    # ---------------- FINAL (prefijo final_) ----------------
    final_features = [f"final_{k}" for k in FEATURE_KEYS]

    train_logreg_for_phase(df, final_features, phase_name="final")

    spark.stop()
