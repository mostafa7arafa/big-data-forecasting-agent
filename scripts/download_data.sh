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

months=()
d="$START-01"
while [[ "${d:0:7}" < "$END" || "${d:0:7}" == "$END" ]]; do
  months+=("${d:0:7}")
  d=$(date -d "$d +1 month" +%F)
done

printf '%s\n' "${months[@]}" |
  xargs -P 6 -I{} bash -c 'fetch "$0/trip-data/yellow_tripdata_{}.parquet" "$1/yellow_tripdata_{}.parquet"' "$BASE" "$DIR"

echo "$(ls "$DIR"/yellow_tripdata_*.parquet | wc -l) monthly files in $DIR"
