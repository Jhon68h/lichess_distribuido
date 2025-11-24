#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os

import chess
import chess.pgn

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

from chess_features import extract_features_from_fen



# Número de movimientos completos (1., 2., 3., ...) que consideramos "apertura"
OPENING_MOVES = 10

# Número de movimientos completos para la FEN de "medio juego"
MIDGAME_MOVES = 20


def create_spark_session(app_name: str = "chess-ml-local") -> SparkSession:
    """Crea la sesión Spark respetando SPARK_MASTER_URL si está definido."""
    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    shuffle_partitions = os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "200")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .getOrCreate()
    )
    print(f">>> Spark master en uso: {spark.sparkContext.master}")
    return spark


def winner_to_label(winner: str):
    """
    Mapea el ganador textual a una etiqueta binaria:
    - "white" → 1
    - "black" o "draw" → 0
    - cualquier otra cosa → None
    """
    if winner is None:
        return None
    w = winner.strip().lower()
    if w == "white":
        return 1
    if w in ("black", "draw"):
        return 0
    return None


def board_fen_after_n_moves(pgn_str: str, n_moves: int) -> str:
    """
    Reproduce la partida desde un PGN y devuelve el FEN tras n_moves
    movimientos completos (es decir, 2 * n_moves plies).
    Si la partida termina antes, devuelve la posición alcanzada.
    """
    if pgn_str is None:
        return None

    pgn_str = str(pgn_str).strip()
    if not pgn_str:
        return None

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
    except Exception:
        return None

    if game is None:
        return None

    board = game.board()
    max_plies = 2 * n_moves
    ply_count = 0

    try:
        for move in game.mainline_moves():
            board.push(move)
            ply_count += 1
            if ply_count >= max_plies:
                break
    except Exception:
        # si se rompe al reproducir, devolvemos la posición alcanzada
        pass

    return board.fen()


def board_final_fen(pgn_str: str) -> str:
    """
    Reproduce la partida completa y devuelve el FEN de la posición final.
    Si hay algún error, devuelve la posición alcanzada hasta el fallo.
    """
    if pgn_str is None:
        return None

    pgn_str = str(pgn_str).strip()
    if not pgn_str:
        return None

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
    except Exception:
        return None

    if game is None:
        return None

    board = game.board()
    try:
        for move in game.mainline_moves():
            board.push(move)
    except Exception:
        # devolvemos la última posición válida
        pass

    return board.fen()


def total_moves_from_pgn(pgn_str: str) -> int:
    """
    Devuelve el número total de movimientos completos de la partida.

    Definición:
    - Se recorre la mainline del PGN.
    - Se cuenta el número de plies (medio-movimientos).
    - total_moves = ceil(plies / 2).

    Si el PGN es inválido o vacío, devuelve None.
    """
    if pgn_str is None:
        return None

    pgn_str = str(pgn_str).strip()
    if not pgn_str:
        return None

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
    except Exception:
        return None

    if game is None:
        return None

    ply_count = 0
    try:
        for _move in game.mainline_moves():
            ply_count += 1
    except Exception:
        # si se rompe al reproducir, contamos hasta donde se pudo
        pass

    if ply_count == 0:
        return 0

    total_moves = (ply_count + 1) // 2
    return int(total_moves)


# UDFs reutilizables
winner_to_label_udf = F.udf(winner_to_label, T.IntegerType())

fen_after_opening_udf = F.udf(
    lambda pgn: board_fen_after_n_moves(pgn, n_moves=OPENING_MOVES),
    T.StringType(),
)

fen_after_20_moves_udf = F.udf(
    lambda pgn: board_fen_after_n_moves(pgn, n_moves=MIDGAME_MOVES),
    T.StringType(),
)

fen_final_udf = F.udf(board_final_fen, T.StringType())

total_moves_udf = F.udf(total_moves_from_pgn, T.IntegerType())


def add_row_index(df, index_col: str = "row_id"):
    """
    Añade un índice incremental para poder alinear datasets por fila.
    """
    window = Window.orderBy(F.monotonically_increasing_id())
    return df.withColumn(index_col, F.row_number().over(window))


def build_base_dataset(
    spark: SparkSession,
    complete_path: str,
    pgn_path: str,
    pgn_col_name: str = "FEN",
    sample_size: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
):
    """
    Construye el dataset base combinando:
    - CSV "completo" con metadatos (Winner, etc.)
    - CSV con las jugadas en formato PGN (en la columna pgn_col_name)

    Devuelve un DataFrame con:
    - columnas originales del CSV completo
    - columna PGN
    - label_white_win
    - fen_after_opening   (apertura)
    - fen_after_20_moves  (medio juego aproximado)
    - fen_final           (posición final)
    - total_moves         (nº total de movimientos de la partida)
    """

    # Carga de los dos CSV
    complete_df = (
        spark.read
        .option("header", True)
        .csv(complete_path)
    )

    pgn_df = spark.read.option("header", True).csv(pgn_path)

    # Solo tomar la primera columna del CSV PGN/FEN (el resto se ignora)
    pgn_first_col = pgn_df.columns[0] if pgn_df.columns else pgn_col_name
    pgn_df = pgn_df.select(pgn_first_col)

    # Alineamos por fila usando un índice artificial
    complete_df = add_row_index(complete_df, index_col="row_id")
    pgn_df = add_row_index(pgn_df, index_col="row_id")

    # Solo usamos la columna con los movimientos, ignorando Site u otras
    pgn_df = pgn_df.select(
        "row_id",
        F.col(pgn_first_col).alias("PGN"),
    )

    df = (
        complete_df
        .join(pgn_df, on="row_id", how="inner")
        .drop("row_id")
    )

    # Filtro básico de calidad: PGN no vacío y que empiece por "1."
    df = df.filter(
        F.col("PGN").isNotNull()
        & (F.length(F.col("PGN")) > 0)
        & F.col("PGN").rlike(r"^\s*1\.")
    )

    # Muestreo opcional por límite de filas
    if sample_size is not None:
        df = df.limit(sample_size)

    # Mezclar filas para mayor variabilidad
    if shuffle:
        df = df.orderBy(F.rand(shuffle_seed))

    # Etiqueta de ganador (blancas ganan vs. resto)
    df = df.withColumn(
        "label_white_win",
        winner_to_label_udf(F.col("Winner")),
    )

    # FEN tras apertura, medio juego y final
    df = df.withColumn(
        "fen_after_opening",
        fen_after_opening_udf(F.col("PGN")),
    )

    df = df.withColumn(
        "fen_after_20_moves",
        fen_after_20_moves_udf(F.col("PGN")),
    )

    df = df.withColumn(
        "fen_final",
        fen_final_udf(F.col("PGN")),
    )

    # Número total de movimientos completos de la partida
    df = df.withColumn(
        "total_moves",
        total_moves_udf(F.col("PGN")),
    )

    return df

FEATURE_KEYS = [
    "material_white",
    "material_black",
    "material_diff",
    "white_pawns",
    "white_doubled_pawns",
    "white_isolated_pawns",
    "white_passed_pawns",
    "white_advanced_pawns",
    "white_pawn_islands",
    "black_pawns",
    "black_doubled_pawns",
    "black_isolated_pawns",
    "black_passed_pawns",
    "black_advanced_pawns",
    "black_pawn_islands",
    "pawns_diff",
    "passed_pawns_diff",
    "advanced_pawns_diff",
    "isolated_pawns_diff",
    "pawn_islands_diff",
    # Geometría rey-peones
    "white_king_pawn_shield",
    "white_king_pawns_near",
    "black_king_pawn_shield",
    "black_king_pawns_near",
    "king_pawn_shield_diff",
    "king_pawns_near_diff",
    # Dispersión de peones
    "white_pawn_file_mean",
    "white_pawn_file_std",
    "white_pawn_rank_mean",
    "white_pawn_rank_std",
    "white_pawn_file_span",
    "white_pawn_rank_span",
    "black_pawn_file_mean",
    "black_pawn_file_std",
    "black_pawn_rank_mean",
    "black_pawn_rank_std",
    "black_pawn_file_span",
    "black_pawn_rank_span",
    "pawn_file_std_diff",
    "pawn_rank_std_diff",
    # Movilidad
    "white_mobility",
    "black_mobility",
    "mobility_diff",
]

FEATURE_SCHEMA = T.StructType(
    [T.StructField(k, T.DoubleType(), True) for k in FEATURE_KEYS]
)


def _fen_features_list(fen: str):
    """
    Aplica extract_features_from_fen y devuelve una lista de valores
    en el mismo orden que FEATURE_KEYS.
    """
    feats = extract_features_from_fen(fen)
    return [feats.get(k) for k in FEATURE_KEYS]


fen_features_udf = F.udf(_fen_features_list, FEATURE_SCHEMA)


def add_positional_features(df, fen_col: str = "fen_after_20_moves"):
    """
    Versión "clásica": extrae features posicionales de la columna FEN indicada
    (por defecto, fen_after_20_moves) y expande las columnas de features
    directamente, SIN prefijo (material_diff, pawns_diff, ...).

    Esta función mantiene compatibilidad con tu código anterior para el midgame.
    """
    df = df.withColumn("features", fen_features_udf(F.col(fen_col)))
    df = df.select("*", "features.*").drop("features")
    return df


def add_positional_features_for_fen(df, fen_col: str, prefix: str):
    """
    Extrae features posicionales desde la columna FEN indicada y
    los añade con un prefijo, por ejemplo:

    prefix = "open"  →  columnas "open_material_diff", "open_pawns_diff", ...
    prefix = "final" →  columnas "final_material_diff", ...
    """
    tmp_col = f"features_{prefix}"
    df = df.withColumn(tmp_col, fen_features_udf(F.col(fen_col)))

    for k in FEATURE_KEYS:
        df = df.withColumn(f"{prefix}_{k}", F.col(f"{tmp_col}.{k}"))

    df = df.drop(tmp_col)
    return df


def add_all_positional_features(df):
    """
    Añade features para las tres fases:

    - Midgame (fen_after_20_moves), SIN prefijo (material_diff, pawns_diff, ...)
    - Apertura (fen_after_opening), con prefijo "open_"
    - Final (fen_final), con prefijo "final_"
    """
    # Midgame (compatibilidad con tu flujo anterior)
    df = add_positional_features(df, fen_col="fen_after_20_moves")

    # Apertura con prefijo "open_"
    df = add_positional_features_for_fen(
        df,
        fen_col="fen_after_opening",
        prefix="open",
    )

    # Final con prefijo "final_"
    df = add_positional_features_for_fen(
        df,
        fen_col="fen_final",
        prefix="final",
    )

    return df
