# -*- coding: utf-8 -*-

"""
Funciones de extracción de features de ajedrez a partir de FEN usando python-chess.

Diseñadas como funciones puras (sin Spark), para luego poder envolverlas en UDFs.
"""

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
# 2. Estructura de peones
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
# 3. Wrapper principal por FEN
# ======================================================================

def extract_features_from_fen(fen: str):
    """
    Extrae un conjunto de features numéricos a partir de una FEN.

    Devuelve un dict con:
    - material_white, material_black, material_diff
    - pawn features para blancas y negras
    - diferencias white - black para algunas magnitudes
    """
    board = safe_board_from_fen(fen)
    if board is None:
        # Puedes devolver None o ceros. Aquí uso None para luego tratarlos como missing.
        return {
            "material_white": None,
            "material_black": None,
            "material_diff": None,
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
        }

    feats = {}

    # Material
    feats.update(material_counts(board))

    # Estructura de peones
    white_pawns = pawn_structure_stats(board, chess.WHITE, prefix="white")
    black_pawns = pawn_structure_stats(board, chess.BLACK, prefix="black")
    feats.update(white_pawns)
    feats.update(black_pawns)

    # Diferencias blancas - negras en algunos stats de peones
    feats["pawns_diff"] = feats["white_pawns"] - feats["black_pawns"]
    feats["passed_pawns_diff"] = feats["white_passed_pawns"] - feats["black_passed_pawns"]
    feats["advanced_pawns_diff"] = feats["white_advanced_pawns"] - feats["black_advanced_pawns"]
    feats["isolated_pawns_diff"] = feats["white_isolated_pawns"] - feats["black_isolated_pawns"]
    feats["pawn_islands_diff"] = feats["white_pawn_islands"] - feats["black_pawn_islands"]

    return feats
