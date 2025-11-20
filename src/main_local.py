#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chess_spark_pipeline import (
    create_spark_session,
    build_base_dataset,
    add_positional_features,
)


if __name__ == "__main__":
    COMPLETE_PATH = "../dataset/Lichess_2013_2014_Complete_sample.csv"
    PGN_PATH = "../dataset/Lichess_2013_2014_FEN_sample.csv"

    # Usar 50 muestras (tu dataset ya está preparado para esto)
    SAMPLE_SIZE = None

    # Ruta donde se guardará el resultado completo
    OUTPUT_PATH = "../experimentos/Lichess_2013_2014_features_full.parquet"

    spark = create_spark_session()
    # Reducir ruido de logs (quita la mayoría de WARN)
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Construir dataset base: metadatos + PGN + label + FEN + total_moves
    df_base = build_base_dataset(
        spark,
        complete_path=COMPLETE_PATH,
        pgn_path=PGN_PATH,
        pgn_col_name="FEN",  # columna con las jugadas "1. e4 e6 2. d4 ..."
        sample_size=SAMPLE_SIZE,
    )

    # 2) Añadir features posicionales (material, peones, geometría, etc.)
    df_feat = add_positional_features(df_base)

    # 3) Mostrar una vista en terminal (similar a como tenías al principio)
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
    ).show(SAMPLE_SIZE, truncate=False)

    # 4) Guardar todos los resultados (todas las columnas) en formato Parquet
    df_feat.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"\nResultados guardados en: {OUTPUT_PATH}\n")

    spark.stop()
