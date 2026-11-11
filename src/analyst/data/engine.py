"""Read-only, bounded SQL access to a configured dataset."""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import Timer

import duckdb
import pyarrow as pa

from analyst.data.dataset import DatasetConfig


class QueryError(Exception):
    """A rejected, failed, or resource-limited query."""


@dataclass(frozen=True)
class Coverage:
    start: datetime
    end: datetime


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class Engine:
    def __init__(
        self,
        dataset: DatasetConfig,
        cutoff: date | None = None,
        *,
        timeout_s: float = 60,
        max_rows: int = 1_000_000,
    ):
        self.cutoff = cutoff
        self._timeout_s = timeout_s
        self._max_rows = max_rows
        self._view_names = [dataset.table, dataset.table + "_raw"] + [
            lookup.name for lookup in dataset.lookups
        ]
        self._table = _identifier(dataset.table)
        self._time_column = _identifier(dataset.time_column)
        self._coverage: Coverage | None = None
        self._con = duckdb.connect(":memory:")
        source = (
            f"read_parquet({_literal(str((dataset.root / dataset.files).resolve()))}, "
            "union_by_name=true)"
        )
        cutoff_filter = (
            f"{self._time_column} < DATE {_literal((cutoff + timedelta(days=1)).isoformat())}"
            if cutoff is not None
            else "TRUE"
        )
        start, end = dataset.valid_time_range
        valid_filter = (
            f"{self._time_column} >= DATE {_literal(start.isoformat())} AND "
            f"{self._time_column} < DATE {_literal(end.isoformat())}"
        )
        self._con.execute(
            f"CREATE VIEW {_identifier(dataset.table + '_raw')} AS "
            f"SELECT * FROM {source} WHERE {cutoff_filter}"
        )
        self._con.execute(
            f"CREATE VIEW {self._table} AS SELECT * FROM {source} "
            f"WHERE {valid_filter} AND {cutoff_filter}"
        )
        for lookup in dataset.lookups:
            path = _literal(str((dataset.root / lookup.file).resolve()))
            self._con.execute(
                f"CREATE VIEW {_identifier(lookup.name)} AS SELECT * FROM read_csv({path})"
            )
        directories = ", ".join(
            _literal(str(path.resolve()) + "/") for path in dataset.data_dirs
        )
        self._con.execute(f"SET allowed_directories=[{directories}]")
        self._con.execute("SET enable_external_access=false")
        self._con.execute("SET memory_limit='8GB'")
        self._con.execute("SET lock_configuration=true")

    def _validate_sql(self, sql: str) -> None:
        message = "query may only read the dataset views: " + ", ".join(self._view_names)
        try:
            # Fully consume this result before executing the user's relation.
            rows = self._con.execute("SELECT json_serialize_sql(?)", [sql]).fetchall()
            ast = json.loads(rows[0][0])
            if ast.get("error"):
                raise QueryError(ast["error_message"].splitlines()[0])

            allowed_views = frozenset(name.casefold() for name in self._view_names)
            pending = [(ast, allowed_views)]
            while pending:
                node, visible_tables = pending.pop()
                if isinstance(node, dict):
                    # Copy scope for this query's subtree, never its siblings/parent.
                    cte_names = []
                    for entry in node.get("cte_map", {}).get("map", []):
                        name = entry["key"]
                        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                            raise QueryError(message)
                        cte_names.append(name.casefold())
                    visible_tables = visible_tables.union(cte_names)

                    if node.get("type") == "BASE_TABLE":
                        name = node["table_name"]
                        if (
                            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                            or name.casefold() not in visible_tables
                            or any(
                                node.get(key, "").casefold() not in {"", "main", "memory"}
                                for key in ("schema_name", "catalog_name")
                            )
                        ):
                            raise QueryError(message)
                    elif (
                        node.get("type") == "TABLE_FUNCTION"
                        and node["function"]["function_name"].casefold()
                        not in {"range", "generate_series", "unnest"}
                    ):
                        raise QueryError(message)
                    pending.extend((child, visible_tables) for child in node.values())
                elif isinstance(node, list):
                    pending.extend((child, visible_tables) for child in node)
        except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
            # Fail closed if the serializer's AST shape is unexpected.
            raise QueryError(message) from exc

    def execute(self, sql: str) -> pa.Table:
        timer = Timer(self._timeout_s, self._con.interrupt)
        timer.daemon = True
        timer.start()
        try:
            statements = self._con.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise QueryError("expected exactly one SELECT statement")
            self._validate_sql(sql)
            result = self._con.sql(sql).limit(self._max_rows + 1).to_arrow_table()
            if result.num_rows > self._max_rows:
                raise QueryError(
                    f"result has more than {self._max_rows} rows; aggregate further"
                )
            return result
        except duckdb.InterruptException as exc:
            raise QueryError(f"query exceeded {self._timeout_s:g}s timeout") from exc
        except duckdb.Error as exc:
            raise QueryError(str(exc).splitlines()[0]) from exc
        finally:
            timer.cancel()
            timer.join()

    def coverage(self) -> Coverage:
        if self._coverage is None:
            result = self.execute(
                f"SELECT min({self._time_column}), max({self._time_column}) "
                f"FROM {self._table}"
            )
            start, end = result[0][0].as_py(), result[1][0].as_py()
            if start is None or end is None:
                raise QueryError("dataset has no rows in the visible time range")
            self._coverage = Coverage(start, end)
        return self._coverage

    def columns(self) -> list[tuple[str, str]]:
        # This engine-generated query only needs relation metadata, not Arrow data.
        relation = self._con.sql(f"SELECT * FROM {self._table} LIMIT 0")
        return list(zip(relation.columns, map(str, relation.types), strict=True))
