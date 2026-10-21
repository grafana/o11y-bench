#!/bin/bash
# Run Grafana once so its database is migrated at build time.
set -e
mkdir -p /dev/shm/grafana
. /etc/grafana/o11y-env.sh
# Migrate only: the datasources are provisioned at boot, as before, so their versions and ids match
export GF_PATHS_PROVISIONING=/tmp/no-provisioning
mkdir -p "$GF_PATHS_PROVISIONING"
/usr/share/grafana/bin/grafana server --homepath=/usr/share/grafana \
    --config=/usr/share/grafana/conf/defaults.ini > /tmp/bake.log 2>&1 &
pid=$!
until curl -sf http://localhost:3000/api/health > /dev/null; do
    kill -0 "$pid" 2> /dev/null || { cat /tmp/bake.log; echo "Grafana exited before it became healthy" >&2; exit 1; }
    sleep 0.1
done
sleep 1
kill "$pid"
wait "$pid" || true
mkdir -p /seed
cp -r /dev/shm/grafana /seed/grafana
