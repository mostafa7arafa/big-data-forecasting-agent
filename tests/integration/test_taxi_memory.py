import pytest

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine
from analyst.memory import load_memory


@pytest.mark.taxi
def test_every_expr_compiles():
    dataset = load_dataset("nyc_taxi")
    model = load_memory(dataset).semantic
    joins = " ".join(
        f"LEFT JOIN {join.table} AS {join.name} ON {join.on}" for join in model.joins
    )
    engine = Engine(dataset)
    items = [*model.dimensions, *model.filters, *model.metrics]
    for item in items:
        engine.execute(f"SELECT {item.expr} FROM {model.table} {joins} LIMIT 0")
