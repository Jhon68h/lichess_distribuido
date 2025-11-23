# Proyecto: Estructuras de Peones y Resultado de Partidas (Spark)

Pipeline local con PySpark para extraer features posicionales desde FEN/PGN de Lichess, generar dataset enriquecido y entrenar un modelo de regresión lineal que predice si ganan blancas.

## Cómo correr todo
1. Requisitos: Python 3.10+, PySpark, python-chess, matplotlib, numpy, scikit-learn.
2. Desde la raíz del repo:
   ```bash
   python src/main_local.py
   ```
   - Para cluster standalone (1 maestro, 3 workers): exporta `SPARK_MASTER_URL=spark://<host-maestro>:7077` antes de ejecutar. El código usará ese master (por defecto local[*]) y fija `spark.sql.shuffle.partitions=200`.

## Qué hace `main_local.py`
- Lee los CSV completos desde `../dataset/` (carpeta hermana al repo).
- Construye FEN tras apertura, jugada 20 (midgame) y final.
- Calcula features de peones/rey/material y movilidad de piezas para las tres fases.
- Limpia FEN inválidos y balancea clases.
- Entrena tres modelos por fase: regresión lineal, HistGradientBoosting (umbral calibrado) y RandomForest (umbral calibrado); compara accuracy/F1/AUC.
- Guarda:
  - Dataset parquet: `experimentos/Lichess_2013_2014_features_full.parquet`
  - Métricas JSON combinadas: `experimentos/model_metrics.json`
  - Gráficas: `experimentos/plots/*.png` (coeficientes, importancias, matrices de confusión)

## Despliegue en Docker/Spark standalone
- Usa el `Dockerfile` y `entrypoint.sh` en cada nodo.
  - Maestro: `docker run -d --name spark-master -e SPARK_MODE=master -p 7077:7077 -p 8080:8080 imagen_spark`
  - Worker: `docker run -d --name spark-worker1 -e SPARK_MODE=worker -e SPARK_MASTER_URL=spark://spark-master:7077 -p 8081:8081 imagen_spark`
  - Driver/pipeline: `docker run --rm -e SPARK_MASTER_URL=spark://spark-master:7077 -v /ruta/datos:/app/dataset -v /ruta/experimentos:/app/experimentos imagen_spark python /app/src/main_local.py`
- Asegúrate de montar los datos en la misma ruta (`/app/dataset`, `/app/experimentos`) para que sean accesibles por maestro/worker/driver.
