#!/usr/bin/env bash
# Download NYC TLC yellow-taxi trip data (Parquet) and the zone lookup into data/raw.
# Usage: scripts/download_data.sh [START=2023-01] [END=2026-09]
set -euo pipefail

START=${1:-2023-01}
END=${2:-2026-09}
BASE=https://d37ci6vzurychx.cloudfront.net
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DIR"

fetch() {  # write to .part, then rename, so readers never see half a file
  local url=$1 out=$2
  [ -s "$out" ] && return 0
  if curl -sfL --retry 5 --retry-all-errors -C - -o "$out.part" "$url"; then
    mv "$out.part" "$out" && echo "ok   $(basename "$out")"
  else
    echo "FAIL $(basename "$out") (rerun to resume)" >&2
    return 1
  fi
}
export -f fetch

fetch "$BASE/misc/taxi_zone_lookup.csv" "$DIR/taxi_zone_lookup.csv"

advance_month() {
  local ym
  ym=$1
  local y=${ym:0:4}
  local m=${ym:5:2}
  m=$((10#$m + 1))
  if (( m > 12 )); then
    m=1
    y=$((y + 1))
  fi
  printf '%04d-%02d' "$y" "$m"
}

months=()
ym="$START"
iter=0
while :; do
  months+=("$ym")
  [[ "$ym" == "$END" ]] && break
  ym=$(advance_month "$ym")
  iter=$((iter + 1))
  if (( iter >= 1000 )); then
    echo "month list exceeded 1000 iterations (START=$START END=$END)" >&2
    exit 1
  fi
done

printf '%s\n' "${months[@]}" |
  xargs -P 6 -I{} bash -c 'fetch "$0/trip-data/yellow_tripdata_{}.parquet" "$1/yellow_tripdata_{}.parquet"' "$BASE" "$DIR"

echo "$(ls "$DIR"/yellow_tripdata_*.parquet | wc -l) monthly files in $DIR"
