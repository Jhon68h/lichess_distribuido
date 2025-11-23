#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejecución local de todo el flujo:
1) Carga de CSVs de Lichess.
2) Construcción de FEN en apertura/medio juego/final + features de peones y movilidad.
3) Entrenamiento de modelos (regresión lineal baseline, HistGradientBoosting y RandomForest) con balance y calibración de umbral.
4) Guardado de métricas, gráficas y dataset enriquecido.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark.sql import functions as F

from chess_spark_pipeline import (
    FEATURE_KEYS,
    add_all_positional_features,
    build_base_dataset,
    create_spark_session,
)
from model_training import (
    train_linear_phase,
    train_hgb_phase,
    train_random_forest_phase,
)


def balance_binary_dataset(df, label_col="label_white_win", ratio=1.0, seed=42):
    """
    Balancea la clase mayoritaria bajando su proporción respecto a la minoritaria.
    ratio=1.0 => deja las clases al 50/50 (aprox).
    """
    counts = {row[label_col]: row["count"] for row in df.groupBy(label_col).count().collect()}
    if len(counts) < 2 or 0 not in counts or 1 not in counts:
        return df  # no hay ambas clases

    minority = 0 if counts[0] < counts[1] else 1
    majority = 1 - minority
    frac_majority = min(1.0, ratio * counts[minority] / counts[majority])
    fractions = {float(minority): 1.0, float(majority): frac_majority}
    return df.sampleBy(label_col, fractions=fractions, seed=seed)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "dataset"
    exp_dir = base_dir.parent / "experimentos"
    plots_dir = exp_dir / "plots"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    complete_path = data_dir / "Lichess_2013_2014_Complete_sample.csv"
    pgn_path = data_dir / "Lichess_2013_2014_FEN_sample.csv"
    output_parquet = exp_dir / "Lichess_2013_2014_features_full.parquet"

    sample_size = None  # usar todo el sample disponible
    n_show = 10

    spark = create_spark_session("chess-ml-local")
    spark.sparkContext.setLogLevel("ERROR")

    print("\n>>> Construyendo dataset base (PGN + FEN + label)...")
    df_base = build_base_dataset(
        spark,
        complete_path=str(complete_path),
        pgn_path=str(pgn_path),
        pgn_col_name="FEN",
        sample_size=sample_size,
        shuffle=True,
        shuffle_seed=42,
    )

    print(">>> Añadiendo features posicionales para apertura/medio/final...")
    df_feat = add_all_positional_features(df_base).cache()
    df_feat = (
        df_feat.filter(
            F.col("fen_after_opening").isNotNull()
            & F.col("fen_after_20_moves").isNotNull()
            & F.col("fen_final").isNotNull()
        )
    ).cache()
    total_rows = df_feat.count()  # materializa
    print(f">>> Filas tras limpiar FEN inválidos: {total_rows}")

    # Balancear clases para evitar colapso a la clase mayoritaria
    df_feat = balance_binary_dataset(df_feat, label_col="label_white_win", ratio=1.0, seed=42).cache()
    balanced_rows = df_feat.count()
    print(f">>> Filas tras balancear clases: {balanced_rows}")

    print("\nVista rápida de midgame (después de 20 jugadas):")
    df_feat.select(
        "label_white_win",
        "total_moves",
        "fen_after_20_moves",
        "material_diff",
        "pawns_diff",
        "passed_pawns_diff",
        "isolated_pawns_diff",
        "king_pawn_shield_diff",
        "pawn_file_std_diff",
        "pawn_rank_std_diff",
    ).show(n_show, truncate=False)

    print(f"\n>>> Guardando dataset enriquecido en {output_parquet}")
    df_feat.write.mode("overwrite").parquet(str(output_parquet))

    # Entrenamiento por fase
    phase_configs = [
        ("midgame_move20", "Medio juego (jugada 20)", FEATURE_KEYS),
        ("opening", "Apertura", [f"open_{k}" for k in FEATURE_KEYS]),
        ("final", "Final", [f"final_{k}" for k in FEATURE_KEYS]),
    ]

    linear_metrics = []
    hgb_metrics = []
    rf_metrics = []
    for phase_name, phase_label, feature_cols in phase_configs:
        print(f"\n>>> Entrenando modelo de regresión lineal para {phase_name}...")
        res_lin = train_linear_phase(
            df_feat,
            feature_cols,
            phase_name,
            plots_dir,
            plot=False,
            phase_label=phase_label,
        )
        linear_metrics.append(res_lin)
        if "error" in res_lin:
            print(f"[{phase_name}] LINEAR ERROR: {res_lin['error']}")
        else:
            print(
                f"[{phase_name}] Linear acc={res_lin['accuracy']:.3f}, "
                f"prec={res_lin['precision']:.3f}, rec={res_lin['recall']:.3f}, rmse={res_lin['rmse']:.3f}"
            )

        print(f">>> Entrenando HistGradientBoosting para {phase_name}...")
        res_log = train_hgb_phase(
            df_feat,
            feature_cols,
            phase_name,
            plots_dir,
            plot=True,
            phase_label=phase_label,
        )
        hgb_metrics.append(res_log)
        if "error" in res_log:
            print(f"[{phase_name}] HGB ERROR: {res_log['error']}")
        else:
            print(
                f"[{phase_name}] HGB auc={res_log.get('auc')}, "
                f"f1={res_log['f1']:.3f}, acc={res_log['accuracy']:.3f}, "
                f"umbral={res_log.get('threshold')}"
            )

        print(f">>> Entrenando Random Forest para {phase_name}...")
        plot_rf = True
        res_rf = train_random_forest_phase(
            df_feat,
            feature_cols,
            phase_name,
            plots_dir,
            plot=plot_rf,
            phase_label=phase_label,
        )
        rf_metrics.append(res_rf)
        if "error" in res_rf:
            print(f"[{phase_name}] RF ERROR: {res_rf['error']}")
        else:
            print(
                f"[{phase_name}] RF auc={res_rf['auc']:.3f}, "
                f"f1={res_rf['f1']:.3f}, acc={res_rf['accuracy']:.3f}, "
                f"umbral={res_rf['threshold']:.2f}"
            )

    metrics_path = exp_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_rows": total_rows,
                "linear_regression": linear_metrics,
                "hist_gradient_boosting": hgb_metrics,
                "random_forest": rf_metrics,
            },
            fh,
            indent=2,
        )
    print(f"\n>>> Métricas guardadas en {metrics_path}")
    print(f">>> Gráficas guardadas en {plots_dir}")

    # -------- Gráfica resumen (1 de 5) --------
    def plot_metric_bar(results, model_name, metric_key, fname):
        labels = [r.get("phase_label", r["phase"]) for r in results if "error" not in r]
        values = [r.get(metric_key) for r in results if "error" not in r]
        plt.figure(figsize=(6, 3))
        plt.bar(labels, values, color="#2563eb")
        plt.title(f"{model_name} - {metric_key.upper()}")
        plt.xlabel("Fase")
        plt.ylabel(metric_key.upper())
        plt.ylim(0, 1)
        plt.tight_layout()
        out = plots_dir / fname
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    bar_path = plot_metric_bar(
        hgb_metrics, "HistGradientBoosting", "accuracy", "resumen_hgb_acc.png"
    )
    print(f">>> Resumen Accuracy HGB: {bar_path}")

    def plot_pawn_feature_effects(df):
        # Medias por resultado para features de peones clave en midgame
        pawn_feats = [
            "pawns_diff",
            "passed_pawns_diff",
            "isolated_pawns_diff",
            "pawn_file_std_diff",
            "pawn_rank_std_diff",
            "king_pawn_shield_diff",
        ]
        summary = (
            df.select(["label_white_win"] + pawn_feats)
            .groupBy("label_white_win")
            .agg(*[F.avg(c).alias(c) for c in pawn_feats])
            .toPandas()
        )
        summary = summary.sort_values("label_white_win")
        plt.figure(figsize=(8, 4))
        for _, row in summary.iterrows():
            label = int(row["label_white_win"])
            vals = [row[c] for c in pawn_feats]
            plt.bar(
                [f"{f}\n(label={label})" for f in pawn_feats],
                vals,
                alpha=0.6 if label == 0 else 0.9,
                label=f"Resultado={label}",
            )
        plt.xticks(rotation=45, ha="right")
        plt.title("Medias de features de peones por resultado (midgame)")
        plt.ylabel("Valor medio")
        plt.legend()
        plt.tight_layout()
        out = plots_dir / "efectos_peones_midgame.png"
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    pawn_plot = plot_pawn_feature_effects(df_feat)
    print(f">>> Efectos de peones guardado en {pawn_plot}")

    spark.stop()


if __name__ == "__main__":
    main()
