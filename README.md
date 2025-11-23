# Proyecto: Estructuras de Peones y Resultado de Partidas (Spark)

Pipeline local con PySpark para extraer features posicionales desde FEN/PGN de Lichess, generar dataset enriquecido y entrenar un modelo de regresión lineal que predice si ganan blancas.

## Cómo correr todo
1. Requisitos: Python 3.10+, PySpark, python-chess, matplotlib, numpy.
2. Desde la raíz del repo:
   ```bash
   python src/main_local.py
   ```

## Qué hace `main_local.py`
- Lee los CSV de `dataset/`.
- Construye FEN tras apertura, jugada 20 (midgame) y final.
- Calcula features de peones/rey/material y movilidad de piezas para las tres fases.
- Limpia FEN inválidos y balancea clases.
- Entrena tres modelos por fase: regresión lineal, HistGradientBoosting (umbral calibrado) y RandomForest (umbral calibrado); compara accuracy/F1/AUC.
- Guarda:
  - Dataset parquet: `experimentos/Lichess_2013_2014_features_full.parquet`
  - Métricas JSON combinadas: `experimentos/model_metrics.json`
  - Gráficas: `experimentos/plots/*.png` (coeficientes, importancias, matrices de confusión)
