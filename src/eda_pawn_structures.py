#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession, functions as F

from chess_spark_pipeline import create_spark_session

def summarize_phase(df, phase_name: str, feature_cols):
    print(f"\n==================== {phase_name.upper()} ====================")

    # Seleccionamos solo label + features de esa fase
    cols = ["label_white_win"] + feature_cols
    df_phase = (
        df.select(cols)
        .dropna()
        .filter(F.col("label_white_win").isin(0, 1))
    )

    n_rows = df_phase.count()
    print(f"Filas válidas: {n_rows}")
    if n_rows == 0:
        print("Sin datos válidos para esta fase.")
        return

    # Distribución de clases
    print("Distribución de clases:")
    df_phase.groupBy("label_white_win").count().show()

    # Medias de cada feature condicionadas al resultado
    agg_exprs = [F.avg(c).alias(c + "_mean") for c in feature_cols]

    stats = (
        df_phase
        .groupBy("label_white_win")
        .agg(*agg_exprs)
        .orderBy("label_white_win")
    )

    print("Medias de features por resultado (label_white_win):")
    stats.show(truncate=False)


if __name__ == "__main__":
    # Ruta del parquet generado por main_local.py
    INPUT_PATH = "../experimentos/Lichess_2013_2014_features_full.parquet"

    spark: SparkSession = create_spark_session("chess-eda-pawns")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(INPUT_PATH)

    # ---------------- MIDGAME (move 20) ----------------
    mid_features = [
        "material_diff",
        "pawns_diff",
        "passed_pawns_diff",
        "isolated_pawns_diff",
        "pawn_file_std_diff",
        "pawn_rank_std_diff",
        "king_pawn_shield_diff",
    ]

    summarize_phase(df, "midgame (move 20)", mid_features)

    # ---------------- OPENING ----------------
    open_features = [
        "open_material_diff",
        "open_pawns_diff",
        "open_passed_pawns_diff",
        "open_isolated_pawns_diff",
        "open_pawn_file_std_diff",
        "open_pawn_rank_std_diff",
        "open_king_pawn_shield_diff",
    ]

    summarize_phase(df, "opening", open_features)

    # ---------------- FINAL ----------------
    final_features = [
        "final_material_diff",
        "final_pawns_diff",
        "final_passed_pawns_diff",
        "final_isolated_pawns_diff",
        "final_pawn_file_std_diff",
        "final_pawn_rank_std_diff",
        "final_king_pawn_shield_diff",
    ]

    summarize_phase(df, "final", final_features)

    spark.stop()
