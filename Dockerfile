FROM debian:bullseye-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3

ENV SPARK_HOME=/opt/spark \
    PATH=$PATH:/opt/spark/bin:/opt/spark/sbin

# Java y utilidades b  sicas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates openjdk-11-jdk bash procps && \
    rm -rf /var/lib/apt/lists/*

# Descargar e instalar Spark precompilado para Hadoop 3
# (URL t  pica de los binarios de Spark 3.5.1):contentReference[oaicite:1]{index=1}
RUN curl -fsSL \
      https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_V>
      -o /tmp/spark.tgz && \
    mkdir -p /opt && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm /tmp/spark.tgz

# Script de entrada que decide si arranca master, worker, etc.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Puertos t  picos de Spark Standalone:
# 7077  -> puerto del master
# 8080  -> Web UI del master
# 8081  -> Web UI de cada worker
# 4040  -> Web UI de jobs (driver)
EXPOSE 7077 8080 8081 4040

ENTRYPOINT ["/entrypoint.sh"]

