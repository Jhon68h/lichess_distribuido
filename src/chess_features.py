# -*- coding: utf-8 -*-

"""
Funciones de extracción de features de ajedrez a partir de FEN usando python-chess.

Diseñadas como funciones puras (sin Spark), para luego poder envolverlas en UDFs.
"""

import math
import chess


# Valores de material estándar
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def safe_board_from_fen(fen: str):
    """
    Intenta construir un chess.Board a partir de un FEN.
    Devuelve None si el FEN es inválido.
    """
    if fen is None:
        return None

    fen = str(fen).strip()
    if not fen:
        return None

    try:
        board = chess.Board(fen)
        return board
    except Exception:
        return None


# ======================================================================
# 1. Material
# ======================================================================

def material_counts(board: chess.Board):
    """
    Cuenta el material por bando y devuelve:
    - total_material_white
    - total_material_black
    - material_diff = white - black
    """
    total_white = 0
    total_black = 0

    for piece_type, value in PIECE_VALUES.items():
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        total_white += white_pieces * value
        total_black += black_pieces * value

    return {
        "material_white": float(total_white),
        "material_black": float(total_black),
        "material_diff": float(total_white - total_black),
    }


# ======================================================================
# 2. Estructura de peones "clásica"
# ======================================================================

def _pawns_by_file(board: chess.Board, color: bool):
    """
    Devuelve un dict {file_index (0-7): [squares]} de peones por columna.
    """
    pawns = board.pieces(chess.PAWN, color)
    pawns_by_file = {f: [] for f in range(8)}
    for sq in pawns:
        file_idx = chess.square_file(sq)  # 0 = 'a', 7 = 'h'
        pawns_by_file[file_idx].append(sq)
    return pawns_by_file


def pawn_structure_stats(board: chess.Board, color: bool, prefix: str):
    """
    Calcula varios features de estructura de peones para un color dado.

    Devuelve un dict con claves prefijadas, por ejemplo:
    - f"{prefix}_pawns"
    - f"{prefix}_doubled_pawns"
    - f"{prefix}_isolated_pawns"
    - f"{prefix}_passed_pawns"
    - f"{prefix}_advanced_pawns"
    - f"{prefix}_pawn_islands"
    """
    pawns = list(board.pieces(chess.PAWN, color))
    pawns_by_file = _pawns_by_file(board, color)

    num_pawns = len(pawns)

    # Peones doblados: archivos con >= 2 peones
    doubled = sum(1 for f in range(8) if len(pawns_by_file[f]) >= 2)

    # Islas de peones: bloques contiguos de columnas con al menos un peón
    files_with_pawns = [f for f in range(8) if len(pawns_by_file[f]) > 0]
    islands = 0
    if files_with_pawns:
        islands = 1
        for i in range(1, len(files_with_pawns)):
            if files_with_pawns[i] != files_with_pawns[i - 1] + 1:
                islands += 1

    # Peones aislados: sin peones en columnas adyacentes
    isolated = 0
    for f in range(8):
        if len(pawns_by_file[f]) == 0:
            continue
        has_left = (f - 1 >= 0) and (len(pawns_by_file[f - 1]) > 0)
        has_right = (f + 1 <= 7) and (len(pawns_by_file[f + 1]) > 0)
        if not has_left and not has_right:
            isolated += len(pawns_by_file[f])

    # Peones pasados y avanzados
    passed = 0
    advanced = 0
    for sq in pawns:
        rank = chess.square_rank(sq)  # 0..7
        file_idx = chess.square_file(sq)

        if color == chess.WHITE:
            # avanzado: 5ª, 6ª, 7ª fila (ranks 4,5,6)
            if rank >= 4:
                advanced += 1

            # peón pasado: no hay peones negros en su columna o adyacentes por delante
            enemy_pawns = board.pieces(chess.PAWN, chess.BLACK)
            is_passed = True
            for ef in range(max(0, file_idx - 1), min(7, file_idx + 1) + 1):
                for er in range(rank + 1, 8):  # filas delante de él
                    sq_enemy = chess.square(ef, er)
                    if sq_enemy in enemy_pawns:
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                passed += 1
        else:
            # BLACK
            # avanzado: 4ª, 3ª, 2ª fila (ranks 3,2,1)
            if rank <= 3:
                advanced += 1

            enemy_pawns = board.pieces(chess.PAWN, chess.WHITE)
            is_passed = True
            for ef in range(max(0, file_idx - 1), min(7, file_idx + 1) + 1):
                for er in range(0, rank):  # filas delante de él para negras (hacia rank 0)
                    sq_enemy = chess.square(ef, er)
                    if sq_enemy in enemy_pawns:
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                passed += 1

    return {
        f"{prefix}_pawns": float(num_pawns),
        f"{prefix}_doubled_pawns": float(doubled),
        f"{prefix}_isolated_pawns": float(isolated),
        f"{prefix}_passed_pawns": float(passed),
        f"{prefix}_advanced_pawns": float(advanced),
        f"{prefix}_pawn_islands": float(islands),
    }


# ======================================================================
# 3. Features geométricos de peones y rey
# ======================================================================

def king_pawn_geometry(board: chess.Board, color: bool, prefix: str):
    """
    Features relacionados con cómo los peones protegen al rey:
    - f"{prefix}_king_pawn_shield": peones en un rectángulo 3x2 delante del rey
      (según la dirección del color).
    - f"{prefix}_king_pawns_near": peones propios en un cuadrado 5x5 alrededor
      del rey (distancia de Chebyshev <= 2).
    """
    king_sq = board.king(color)
    if king_sq is None:
        return {
            f"{prefix}_king_pawn_shield": 0.0,
            f"{prefix}_king_pawns_near": 0.0,
        }

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    own_pawns = board.pieces(chess.PAWN, color)

    # 1) Peones cercanos al rey (ventana 5x5, distancia Chebyshev <= 2)
    pawns_near = 0
    for sq in own_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if max(abs(f - king_file), abs(r - king_rank)) <= 2:
            pawns_near += 1

    # 2) Escudo de peones delante del rey
    # Para blancas: filas superiores (rank +1, +2)
    # Para negras: filas inferiores (rank -1, -2)
    shield = 0
    if color == chess.WHITE:
        rank_range = [king_rank + 1, king_rank + 2]
    else:
        rank_range = [king_rank - 1, king_rank - 2]

    file_range = [king_file - 1, king_file, king_file + 1]

    for f in file_range:
        if f < 0 or f > 7:
            continue
        for r in rank_range:
            if r < 0 or r > 7:
                continue
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                shield += 1

    return {
        f"{prefix}_king_pawn_shield": float(shield),
        f"{prefix}_king_pawns_near": float(pawns_near),
    }


def pawn_dispersion(board: chess.Board, color: bool, prefix: str):
    """
    Mide la dispersión geométrica de los peones de un color:

    - f"{prefix}_pawn_file_mean": media de columna (0 = 'a', 7 = 'h')
    - f"{prefix}_pawn_file_std": desviación estándar de columnas
    - f"{prefix}_pawn_rank_mean": media de fila (0 = 1ª, 7 = 8ª)
    - f"{prefix}_pawn_rank_std": desviación estándar de filas
    - f"{prefix}_pawn_file_span": max(file) - min(file)
    - f"{prefix}_pawn_rank_span": max(rank) - min(rank)

    Si no hay peones, devuelve todo 0.0 (posición muy "dispersa" en el sentido
    de que no hay estructura de peones).
    """
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return {
            f"{prefix}_pawn_file_mean": 0.0,
            f"{prefix}_pawn_file_std": 0.0,
            f"{prefix}_pawn_rank_mean": 0.0,
            f"{prefix}_pawn_rank_std": 0.0,
            f"{prefix}_pawn_file_span": 0.0,
            f"{prefix}_pawn_rank_span": 0.0,
        }

    files = [chess.square_file(sq) for sq in pawns]
    ranks = [chess.square_rank(sq) for sq in pawns]

    n = float(len(pawns))

    file_mean = sum(files) / n
    rank_mean = sum(ranks) / n

    file_var = sum((f - file_mean) ** 2 for f in files) / n
    rank_var = sum((r - rank_mean) ** 2 for r in ranks) / n

    file_std = math.sqrt(file_var)
    rank_std = math.sqrt(rank_var)

    file_span = max(files) - min(files)
    rank_span = max(ranks) - min(ranks)

    return {
        f"{prefix}_pawn_file_mean": float(file_mean),
        f"{prefix}_pawn_file_std": float(file_std),
        f"{prefix}_pawn_rank_mean": float(rank_mean),
        f"{prefix}_pawn_rank_std": float(rank_std),
        f"{prefix}_pawn_file_span": float(file_span),
        f"{prefix}_pawn_rank_span": float(rank_span),
    }


# ======================================================================
# 4. Wrapper principal por FEN
# ======================================================================

def extract_features_from_fen(fen: str):
    """
    Extrae un conjunto de features numéricos a partir de una FEN.

    Devuelve un dict con:
    - material_white, material_black, material_diff
    - pawn features clásicas para blancas y negras
    - diferencias white - black para algunas magnitudes de peones
    - features geométricos:
        * escudo de peones y peones cercanos al rey por color
        * dispersión de peones por columnas/filas por color
        * diferencias blanco - negro en algunos de estos rasgos geométricos
    - movilidad (actividad de piezas): número de movimientos legales por color
    """
    board = safe_board_from_fen(fen)
    if board is None:
        # Todos los features a None si el FEN es inválido
        return {
            # Material
            "material_white": None,
            "material_black": None,
            "material_diff": None,
            # Peones clásicos
            "white_pawns": None,
            "white_doubled_pawns": None,
            "white_isolated_pawns": None,
            "white_passed_pawns": None,
            "white_advanced_pawns": None,
            "white_pawn_islands": None,
            "black_pawns": None,
            "black_doubled_pawns": None,
            "black_isolated_pawns": None,
            "black_passed_pawns": None,
            "black_advanced_pawns": None,
            "black_pawn_islands": None,
            "pawns_diff": None,
            "passed_pawns_diff": None,
            "advanced_pawns_diff": None,
            "isolated_pawns_diff": None,
            "pawn_islands_diff": None,
            # Geometría rey-peones
            "white_king_pawn_shield": None,
            "white_king_pawns_near": None,
            "black_king_pawn_shield": None,
            "black_king_pawns_near": None,
            "king_pawn_shield_diff": None,
            "king_pawns_near_diff": None,
            # Dispersión de peones
            "white_pawn_file_mean": None,
            "white_pawn_file_std": None,
            "white_pawn_rank_mean": None,
            "white_pawn_rank_std": None,
            "white_pawn_file_span": None,
            "white_pawn_rank_span": None,
            "black_pawn_file_mean": None,
            "black_pawn_file_std": None,
            "black_pawn_rank_mean": None,
            "black_pawn_rank_std": None,
            "black_pawn_file_span": None,
            "black_pawn_rank_span": None,
            "pawn_file_std_diff": None,
            "pawn_rank_std_diff": None,
            # Movilidad
            "white_mobility": None,
            "black_mobility": None,
            "mobility_diff": None,
        }

    feats = {}

    # Material
    feats.update(material_counts(board))

    # Estructura clásica de peones
    white_pawns = pawn_structure_stats(board, chess.WHITE, prefix="white")
    black_pawns = pawn_structure_stats(board, chess.BLACK, prefix="black")
    feats.update(white_pawns)
    feats.update(black_pawns)

    # Geometría rey-peones
    white_king_geom = king_pawn_geometry(board, chess.WHITE, prefix="white")
    black_king_geom = king_pawn_geometry(board, chess.BLACK, prefix="black")
    feats.update(white_king_geom)
    feats.update(black_king_geom)

    # Dispersión de peones
    white_disp = pawn_dispersion(board, chess.WHITE, prefix="white")
    black_disp = pawn_dispersion(board, chess.BLACK, prefix="black")
    feats.update(white_disp)
    feats.update(black_disp)

    # Movilidad: número de jugadas legales si mueve cada color
    white_mobility = chess.Board(board.fen())  # copiar posición
    white_mobility.turn = chess.WHITE
    black_mobility = chess.Board(board.fen())
    black_mobility.turn = chess.BLACK

    w_mob = float(white_mobility.legal_moves.count())
    b_mob = float(black_mobility.legal_moves.count())

    feats["white_mobility"] = w_mob
    feats["black_mobility"] = b_mob
    feats["mobility_diff"] = w_mob - b_mob

    # Diferencias blancas - negras en algunos stats de peones (clásicos)
    feats["pawns_diff"] = feats["white_pawns"] - feats["black_pawns"]
    feats["passed_pawns_diff"] = feats["white_passed_pawns"] - feats["black_passed_pawns"]
    feats["advanced_pawns_diff"] = feats["white_advanced_pawns"] - feats["black_advanced_pawns"]
    feats["isolated_pawns_diff"] = feats["white_isolated_pawns"] - feats["black_isolated_pawns"]
    feats["pawn_islands_diff"] = feats["white_pawn_islands"] - feats["black_pawn_islands"]

    # Diferencias blancas - negras en rasgos geométricos
    feats["king_pawn_shield_diff"] = (
        feats["white_king_pawn_shield"] - feats["black_king_pawn_shield"]
    )
    feats["king_pawns_near_diff"] = (
        feats["white_king_pawns_near"] - feats["black_king_pawns_near"]
    )
    feats["pawn_file_std_diff"] = (
        feats["white_pawn_file_std"] - feats["black_pawn_file_std"]
    )
    feats["pawn_rank_std_diff"] = (
        feats["white_pawn_rank_std"] - feats["black_pawn_rank_std"]
    )

    return feats
