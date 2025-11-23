#!/usr/bin/env bash
set -e

: "${SPARK_HOME:=/opt/spark}"

MODE="${SPARK_MODE:-shell}"

case "$MODE" in
  master)
    : "${SPARK_MASTER_HOST:=$(hostname)}"
    : "${SPARK_MASTER_PORT:=7077}"
    : "${SPARK_MASTER_WEBUI_PORT:=8080}"
    # Direcci  n de bind; por defecto la misma que SPARK_MASTER_HOST
    : "${SPARK_LOCAL_IP:=${SPARK_MASTER_HOST}}"

    export SPARK_MASTER_HOST SPARK_MASTER_PORT SPARK_MASTER_WEBUI_PORT SPARK_LOCAL_IP
    export SPARK_NO_DAEMONIZE=true

    # No pasamos --host/--port por argumentos, se toman de las variables de entorno
    exec "${SPARK_HOME}/sbin/start-master.sh"
    ;;
  worker)
    # Debes pasar SPARK_MASTER_URL: spark://<ip-master>:7077
    : "${SPARK_MASTER_URL:?Debe definir SPARK_MASTER_URL (por ej. spark://10.5.7.245:7077)}"
    : "${SPARK_WORKER_CORES:=$(nproc)}"
    : "${SPARK_WORKER_MEMORY:=4G}"
    : "${SPARK_WORKER_WEBUI_PORT:=8081}"

    export SPARK_NO_DAEMONIZE=true
    exec "${SPARK_HOME}/sbin/start-worker.sh" \
         --cores "${SPARK_WORKER_CORES}" \
         --memory "${SPARK_WORKER_MEMORY}" \
         --webui-port "${SPARK_WORKER_WEBUI_PORT}" \
         "${SPARK_MASTER_URL}"
    ;;

  submit)
    shift || true
    exec "${SPARK_HOME}/bin/spark-submit" "$@"
    ;;

  shell)
    exec "${SPARK_HOME}/bin/spark-shell"
    ;;

  pyshell)
    exec "${SPARK_HOME}/bin/pyspark"
    ;;

  *)
    exec "$@"
    ;;
esac
