#!/usr/bin/env bash
# RyuDB container entrypoint. The env vars (RYUDB_HOST / RYUDB_PORT /
# RYUDB_PG_PORT / RYUDB_DATA) are set in the Dockerfile; `ryudb-server` reads
# them, so the default `docker run ryudb` just starts the server.
#
# A known command as the first arg is exec'd directly so these all work:
#   docker run ryudb                                   -> ryudb-server (defaults)
#   docker run ryudb --data /data --port 6000           -> ryudb-server <flags>
#   docker run -it ryudb ryudb -e "SELECT ..."         -> the REPL/CLI
#   docker run -it ryudb bash                           -> a shell
set -euo pipefail

case "${1:-}" in
  ryudb|ryudb-server|bash|sh|python|python3)
    exec "$@"
    ;;
  *)
    exec ryudb-server "$@"
    ;;
esac