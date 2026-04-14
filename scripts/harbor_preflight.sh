#!/usr/bin/env bash
set -euo pipefail

# Clean stale Harbor Docker compose projects and pre-build shared images.

CLEANUP_ONLY=0
if [ "${1:-}" = "--cleanup-only" ]; then
  CLEANUP_ONLY=1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker not available; skipping Harbor preflight"
  exit 0
fi

DOCKER_FMT='{{.Label "com.docker.compose.project"}}\t{{.Label "com.docker.compose.service"}}\t{{.State}}\t{{.Label "com.docker.compose.project.config_files"}}'
HARBOR_CONFIG_PATTERN='harbor/environments/docker/docker-compose-base\.yaml'

remove_container_ids() {
  local ids="$1"
  local attempt=1
  local max_attempts=4

  while [ "$attempt" -le "$max_attempts" ]; do
    if [ -z "$ids" ]; then
      return 0
    fi

    local output
    if output=$(docker rm -f $ids 2>&1); then
      return 0
    fi

    if printf '%s' "$output" | grep -qi "already in progress"; then
      sleep $attempt
      attempt=$((attempt + 1))
      continue
    fi

    printf '%s\n' "$output" >&2
    return 1
  done

  local remaining_ids=""
  local id
  for id in $ids; do
    if [ -n "$(docker ps -aq --filter "id=$id")" ]; then
      remaining_ids="${remaining_ids}${remaining_ids:+ }$id"
    fi
  done

  if [ -n "$remaining_ids" ]; then
    echo "Failed to remove Docker containers: $remaining_ids" >&2
    return 1
  fi

  return 0
}

harbor_project_count() {
  docker ps --format "$DOCKER_FMT" |
    awk -F'\t' -v pat="$HARBOR_CONFIG_PATTERN" '$1 != "" && $4 ~ pat { p[$1]=1 } END { print length(p)+0 }'
}

harbor_controller_count() {
  (pgrep -f "harbor run" 2>/dev/null || true) | wc -l | tr -d " "
}

before=$(docker ps -q | wc -l | tr -d " ")
before_projects=$(harbor_project_count)
echo "Docker containers before: $before (Harbor compose projects running: $before_projects)"

controller_count=$(harbor_controller_count)
if [ "$controller_count" -eq 0 ]; then
  # No Harbor controllers are running, so any surviving Harbor compose project is stale.
  stale_projects=$(docker ps -a --format "$DOCKER_FMT" |
    awk -F'\t' -v pat="$HARBOR_CONFIG_PATTERN" '$1 != "" && $4 ~ pat { projects[$1]=1 } END { for (p in projects) print p }'
  )
else
  # During an active suite run, only reap projects whose main container is already gone.
  stale_projects=$(docker ps -a --format "$DOCKER_FMT" |
    awk -F'\t' -v pat="$HARBOR_CONFIG_PATTERN" '
      $1 != "" && $4 ~ pat { projects[$1]=1; if ($2 == "main" && $3 == "running") running[$1]=1 }
      END { for (p in projects) if (!(p in running)) print p }
    '
  )
fi

while IFS= read -r project; do
  [ -n "$project" ] || continue
  echo "Removing stale Harbor project: $project"
  ids=$(docker ps -aq --filter "label=com.docker.compose.project=$project")
  if [ -n "$ids" ]; then
    # shellcheck disable=SC2086
    remove_container_ids "$ids"
  fi
done <<< "$stale_projects"

if [ "$CLEANUP_ONLY" -eq 1 ]; then
  after=$(docker ps -q | wc -l | tr -d " ")
  after_projects=$(harbor_project_count)
  echo "Docker containers after cleanup: $after (Harbor compose projects running: $after_projects)"
  exit 0
fi

docker network inspect o11y-bench-shared >/dev/null 2>&1 || docker network create o11y-bench-shared >/dev/null

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building shared Harbor main image: o11y-bench-main:latest"
docker build -t o11y-bench-main:latest -f "$ROOT/environment/Dockerfile" "$ROOT/environment" >/dev/null
echo "Building shared Harbor sidecar image: o11y-bench-o11y-stack:latest"
docker build -t o11y-bench-o11y-stack:latest -f "$ROOT/docker/Dockerfile" "$ROOT/docker" >/dev/null

after=$(docker ps -q | wc -l | tr -d " ")
after_projects=$(harbor_project_count)
echo "Docker containers after: $after (Harbor compose projects running: $after_projects)"
