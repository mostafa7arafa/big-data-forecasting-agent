FROM python:3.12-slim

# Versions pinned from uv.lock (polars and plotly are not in the lock; latest verified 2026-11-14).
RUN pip install --no-cache-dir \
    duckdb==2.0.0 pandas==3.1.1 polars==2.1.0 numpy==2.5.4 scipy==1.18.1 \
    statsmodels==0.15.0 statsforecast==2.0.1 pyarrow==26.0.1 numba==0.68.1 \
    plotly==7.1.0

# Runs with a read-only root, tmpfs /tmp and an arbitrary --user uid:gid, so nothing
# may depend on a specific uid: HOME and caches live in /tmp, /srv is world-readable.
ENV HOME=/tmp NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1
COPY --chmod=0644 server.py /srv/server.py

USER 10001
WORKDIR /tmp
ENTRYPOINT ["python", "-u", "/srv/server.py"]
