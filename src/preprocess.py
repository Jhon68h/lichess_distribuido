#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io

import chess
import chess.pgn

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

from chess_features import extract_features_from_fen


def create_spark_session(app_name: str = "chess-ml-local") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


def winner_to_label(winner: str):
    if winner is None:
        return None
    w = winner.strip().lower()
    if w == "white":
        return 1
    if w in ("black", "draw"):
        return 0
    return None


def board_fen_after_n_moves(pgn_str: str, n_moves: int) -> str:
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


winner_to_label_udf = F.udf(winner_to_label, T.IntegerType())

fen_after_20_moves_udf = F.udf(
    lambda pgn: board_fen_after_n_moves(pgn, n_moves=20),
    T.StringType(),
)


def add_row_index(df, index_col: str = "row_id"):
    window = Window.orderBy(F.monotonically_increasing_id())
    return df.withColumn(index_col, F.row_number().over(window))


def build_base_dataset(
    spark: SparkSession,
    complete_path: str,
    pgn_path: str,
    pgn_col_name: str = "FEN",
    sample_size: int = 20,
):
    complete_df = (
        spark.read
        .option("header", True)
        .csv(complete_path)
    )

    pgn_df = (
        spark.read
        .option("header", True)
        .csv(pgn_path)
    )

    # alineamos por fila usando un índice artificial
    complete_df = add_row_index(complete_df, index_col="row_id")
    pgn_df = add_row_index(pgn_df, index_col="row_id")

    # solo usamos la columna con los movimientos, ignorando Site
    pgn_df = pgn_df.select(
        "row_id",
        F.col(pgn_col_name).alias("PGN"),
    )

    df = (
        complete_df
        .join(pgn_df, on="row_id", how="inner")
        .drop("row_id")
    )

    # filtro básico de calidad: PGN no vacío y que empiece por "1."
    df = df.filter(
        F.col("PGN").isNotNull()
        & (F.length(F.col("PGN")) > 0)
        & F.col("PGN").rlike(r"^\s*1\.")
    )

    if sample_size is not None:
        df = df.limit(sample_size)

    df = df.withColumn(
        "label_white_win",
        winner_to_label_udf(F.col("Winner")),
    )

    df = df.withColumn(
        "fen_after_20_moves",
        fen_after_20_moves_udf(F.col("PGN")),
    )

    return df


# -------- features posicionales en Spark --------

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
]

FEATURE_SCHEMA = T.StructType(
    [T.StructField(k, T.DoubleType(), True) for k in FEATURE_KEYS]
)


def _fen_features_list(fen: str):
    feats = extract_features_from_fen(fen)
    return [feats[k] for k in FEATURE_KEYS]


fen_features_udf = F.udf(_fen_features_list, FEATURE_SCHEMA)


def add_positional_features(df):
    df = df.withColumn("features", fen_features_udf(F.col("fen_after_20_moves")))
    df = df.select("*", "features.*").drop("features")
    return df


if __name__ == "__main__":
    COMPLETE_PATH = "../dataset/Lichess_2013_2014_Complete_sample.csv"
    PGN_PATH = "../dataset/Lichess_2013_2014_FEN_sample.csv"

    spark = create_spark_session()

    df_base = build_base_dataset(
        spark,
        complete_path=COMPLETE_PATH,
        pgn_path=PGN_PATH,
        pgn_col_name="FEN",  # columna con las jugadas "1. e4 e6 2. d4 ..."
        sample_size=20,
    )

    df_feat = add_positional_features(df_base)

    df_feat.select(
        "WhiteElo",
        "BlackElo",
        "label_white_win",
        "fen_after_20_moves",
        "material_diff",
        "pawns_diff",
        "passed_pawns_diff",
        "isolated_pawns_diff",
    ).show(20, truncate=False)

    spark.stop()
