#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrenamiento y evaluación de modelos Spark ML para predecir si ganan blancas:
- Regresión lineal (como regresor + umbral)
- Regresión logística
- Random Forest

Incluye balanceo de clases, búsqueda de umbral y generación de gráficas.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, functions as F


def _prepare_phase_df(df: DataFrame, feature_cols: Iterable[str]) -> DataFrame:
    """Filtra filas válidas y deja solo label + features."""
    cols = ["label_white_win"] + list(feature_cols)
    return (
        df.select(cols)
        .dropna()
        .filter(F.col("label_white_win").isin(0, 1))
    )


def _add_class_weights(df: DataFrame, label_col: str = "label_white_win"):
    """
    Devuelve df con columna 'class_weight' para balancear clases
    (positiva = número de negativos / número de positivos).
    """
    pos = df.filter(F.col(label_col) == 1).count()
    neg = df.filter(F.col(label_col) == 0).count()
    if pos == 0:
        weight_pos = 1.0
    else:
        weight_pos = neg / pos if pos > 0 else 1.0

    return df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == 1, F.lit(float(weight_pos))).otherwise(F.lit(1.0)),
    )


def _classification_metrics(pred_df: DataFrame) -> dict:
    """Calcula métricas simples a partir de pred_df con cols label_white_win, pred_label."""
    label_col = F.col("label_white_win")
    pred_col = F.col("pred_label")

    tp = pred_df.filter((label_col == 1) & (pred_col == 1)).count()
    tn = pred_df.filter((label_col == 0) & (pred_col == 0)).count()
    fp = pred_df.filter((label_col == 0) & (pred_col == 1)).count()
    fn = pred_df.filter((label_col == 1) & (pred_col == 0)).count()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _plot_coefficients(feature_names, weights, title: str, output_path: Path):
    """Guarda gráfico de barras con los 15 coeficientes de mayor magnitud."""
    coeffs = list(zip(feature_names, weights))
    coeffs = sorted(coeffs, key=lambda x: abs(x[1]), reverse=True)[:15]
    names, vals = zip(*coeffs) if coeffs else ([], [])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(vals)), vals, color="#3b82f6")
    plt.xticks(range(len(vals)), names, rotation=45, ha="right")
    plt.title(title)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    # Etiquetas simples encima de las barras
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_confusion(metrics: dict, title: str, output_path: Path):
    """Pinta matriz de confusión 2x2."""
    tp = metrics.get("tp", 0)
    tn = metrics.get("tn", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)

    mat = np.array([[tn, fp], [fn, tp]], dtype=float)

    plt.figure(figsize=(4, 3.5))
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Real 0", "Real 1"])
    plt.title(title)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(mat[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_feature_importances(feature_names, importances, title: str, output_path: Path):
    """Gráfico de barras con importancias (ej. Random Forest)."""
    pairs = list(zip(feature_names, importances))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:15]
    names, vals = zip(*pairs) if pairs else ([], [])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(vals)), vals, color="#10b981")
    plt.xticks(range(len(vals)), names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()

    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _evaluate_thresholds(prob_df: DataFrame, thresholds: list[float]):
    """
    Evalúa varias probabilidades → etiquetas usando distintos umbrales
    y devuelve la mejor según F1 junto con el listado completo.
    """
    results = []
    for th in thresholds:
        evaluated = prob_df.withColumn(
            "pred_label",
            F.when(F.col("prob1") >= F.lit(th), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        metrics = _classification_metrics(evaluated)
        metrics["threshold"] = th
        results.append(metrics)

    if not results:
        return None, []

    best = max(results, key=lambda m: m.get("f1", 0.0))
    return best, results


def train_linear_phase(
    df: DataFrame,
    feature_cols: list[str],
    phase_name: str,
    plots_dir: Path,
    plot: bool = False,
) -> dict:
    """
    Entrena regresión lineal para una fase concreta y devuelve métricas + coeficientes.

    - Usa VectorAssembler + StandardScaler + LinearRegression (Spark).
    - Predicción binaria: pred_label = 1 si prediction >= 0.5.
    - Guarda gráficos de coeficientes y matriz de confusión en plots_dir.
    """
    df_phase = _prepare_phase_df(df, feature_cols)
    n_rows = df_phase.count()

    if n_rows == 0:
        return {"phase": phase_name, "rows": 0, "error": "sin filas válidas"}

    train_df, test_df = df_phase.randomSplit([0.8, 0.2], seed=42)
    if train_df.count() == 0 or test_df.count() == 0:
        return {"phase": phase_name, "rows": n_rows, "error": "train/test vacíos"}

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
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label_white_win",
        predictionCol="prediction",
        maxIter=50,
        regParam=0.1,
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)
    predictions = predictions.withColumn(
        "pred_label",
        F.when(F.col("prediction") >= 0.5, F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    # Métricas de clasificación
    metrics = _classification_metrics(predictions)

    # Métricas regresivas básicas sobre la salida continua
    reg_eval = RegressionEvaluator(
        labelCol="label_white_win",
        predictionCol="prediction",
        metricName="rmse",
    )
    metrics["rmse"] = reg_eval.evaluate(predictions)

    reg_eval_r2 = RegressionEvaluator(
        labelCol="label_white_win",
        predictionCol="prediction",
        metricName="r2",
    )
    metrics["r2"] = reg_eval_r2.evaluate(predictions)

    weights = model.stages[-1].coefficients.toArray().tolist()
    intercept = float(model.stages[-1].intercept)

    # Graficar
    coef_plot_path = None
    if plot:
        coef_plot_path = plots_dir / f"{phase_name}_coeficientes.png"
        _plot_coefficients(feature_cols, weights, f"{phase_name}: pesos", coef_plot_path)

    metrics.update(
        {
            "phase": phase_name,
            "rows": n_rows,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "intercept": intercept,
            "coef_plot": str(coef_plot_path) if coef_plot_path else None,
        }
    )

    # Top coeficientes ordenados por magnitud
    coef_sorted = sorted(
        zip(feature_cols, weights),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    metrics["top_coefficients"] = [
        {"feature": name, "weight": weight} for name, weight in coef_sorted[:15]
    ]

    return metrics


def train_logistic_phase(
    df: DataFrame,
    feature_cols: list[str],
    phase_name: str,
    plots_dir: Path,
    thresholds: list[float] | None = None,
    plot: bool = False,
) -> dict:
    """
    Entrena regresión logística con balance de clases y búsqueda de umbral.
    Devuelve métricas (AUC, F1, accuracy, etc.) y gráfico de coeficientes.
    """
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

    df_phase = _prepare_phase_df(df, feature_cols)
    n_rows = df_phase.count()
    if n_rows == 0:
        return {"phase": phase_name, "rows": 0, "error": "sin filas válidas"}

    df_phase = _add_class_weights(df_phase)

    train_df, test_df = df_phase.randomSplit([0.8, 0.2], seed=42)
    if train_df.count() == 0 or test_df.count() == 0:
        return {"phase": phase_name, "rows": n_rows, "error": "train/test vacíos"}

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    lr = SparkLogisticRegression(
        featuresCol="features",
        labelCol="label_white_win",
        weightCol="class_weight",
        maxIter=200,
        regParam=0.05,
        elasticNetParam=0.0,
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)
    preds = preds.withColumn("prob1", vector_to_array(F.col("probability")).getItem(1))

    # AUC base
    evaluator = BinaryClassificationEvaluator(
        labelCol="label_white_win",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator.evaluate(preds)

    best_metrics_raw, all_thresholds = _evaluate_thresholds(preds, thresholds)
    if best_metrics_raw is None:
        return {"phase": phase_name, "rows": n_rows, "error": "sin métricas"}
    # Copias para evitar referencias circulares
    best_metrics = dict(best_metrics_raw)
    thresholds_list = [dict(m) for m in all_thresholds]

    # Graficar coeficientes
    weights = model.stages[-1].coefficients.toArray().tolist()
    coef_plot_path = None
    if plot:
        coef_plot_path = plots_dir / f"{phase_name}_logistic_coef.png"
        _plot_coefficients(feature_cols, weights, f"{phase_name}: pesos logística", coef_plot_path)

    coef_sorted = sorted(
        zip(feature_cols, weights),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    best_metrics.update(
        {
            "phase": phase_name,
            "rows": n_rows,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "auc": auc,
            "coef_plot": str(coef_plot_path) if coef_plot_path else None,
            "top_coefficients": [
                {"feature": n, "weight": w} for n, w in coef_sorted[:15]
            ],
            "thresholds_scanned": thresholds_list,
        }
    )
    return best_metrics


def train_random_forest_phase(
    df: DataFrame,
    feature_cols: list[str],
    phase_name: str,
    plots_dir: Path,
    thresholds: list[float] | None = None,
    plot: bool = False,
) -> dict:
    """
    Entrena un RandomForestClassifier con balance de clases y búsqueda de umbral.
    Devuelve métricas (AUC, F1, accuracy, etc.) y gráfico de importancias.
    """
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

    df_phase = _prepare_phase_df(df, feature_cols)
    n_rows = df_phase.count()
    if n_rows == 0:
        return {"phase": phase_name, "rows": 0, "error": "sin filas válidas"}

    df_phase = _add_class_weights(df_phase)

    train_df, test_df = df_phase.randomSplit([0.8, 0.2], seed=42)
    if train_df.count() == 0 or test_df.count() == 0:
        return {"phase": phase_name, "rows": n_rows, "error": "train/test vacíos"}

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_white_win",
        weightCol="class_weight",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=60,
        maxDepth=8,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, rf])
    model = pipeline.fit(train_df)

    preds = (
        model.transform(test_df)
        .withColumn("prob1", vector_to_array(F.col("probability")).getItem(1))
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label_white_win",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator.evaluate(preds)

    best_metrics_raw, all_thresholds = _evaluate_thresholds(preds, thresholds)
    if best_metrics_raw is None:
        return {"phase": phase_name, "rows": n_rows, "error": "sin métricas"}
    best_metrics = dict(best_metrics_raw)
    thresholds_list = [dict(m) for m in all_thresholds]

    importances = model.stages[-1].featureImportances.toArray().tolist()
    imp_plot_path = None
    if plot:
        imp_plot_path = plots_dir / f"{phase_name}_rf_importances.png"
        _plot_feature_importances(feature_cols, importances, f"{phase_name}: RF importancias", imp_plot_path)

    best_metrics.update(
        {
            "phase": phase_name,
            "rows": n_rows,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "auc": auc,
            "importances_plot": str(imp_plot_path) if imp_plot_path else None,
            "thresholds_scanned": thresholds_list,
        }
    )

    # Top importancias
    pairs = list(zip(feature_cols, importances))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:15]
    best_metrics["top_importances"] = [
        {"feature": n, "importance": v} for n, v in pairs
    ]

    return best_metrics
