from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("inspect-chess-parquet")
    .master("local[*]")
    .getOrCreate()
)

OUTPUT_PATH = "../dataset/Lichess_2013_2014_features_50.parquet"

df = spark.read.parquet(OUTPUT_PATH)

# Ver el esquema (todas las columnas, tipos)
df.printSchema()

# Ver las primeras filas
df.show(10, truncate=False)
