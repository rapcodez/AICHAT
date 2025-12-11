import os
import pathlib
import sys
import pytest

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
os.environ["DISABLE_LLM"] = "true"

import app


def test_find_item_and_inventory():
    item = app.find_item("BMS0001")
    assert item is not None
    stock = app.get_inventory("BMS0001")
    assert not stock.empty
    assert set(stock["location_code"]).issuperset({"CHI_WH1", "DAL_WH1"})


def test_forecast_builds_with_brand_and_family():
    forecast_df, stats = app.build_forecast(brand="Cummins", item_family="ISX")
    assert "forecast" in forecast_df["type"].values
    assert stats["label"]


def test_create_and_check_order():
    record = {
        "customer_name": "Acme",
        "ship_to_city": "Dallas",
        "ship_to_country": "USA",
        "item_code": "BMS0002",
        "order_qty": 3,
    }
    order = app.create_order(record)
    fetched = app.check_order_status(order["order_id"])
    assert fetched is not None
    assert fetched["item_code"] == "BMS0002"


def test_handle_inventory_intent_no_llm():
    response, pdf_path, _ = app.handle_intent("Inventory check for BMS0003", {}, [])
    assert "BMS0003" in response
    assert pdf_path is None


def test_respond_with_model_stubbed():
    reply = app.respond_with_model("Hello", [])
    assert "stubbed" in reply
