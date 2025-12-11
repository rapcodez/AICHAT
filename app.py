import datetime
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fpdf import FPDF
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ----------------------
# Data seeding
# ----------------------

ITEMS = [
    {
        "item_code": "BMS0001",
        "item_name": "Fleetguard Oil Filter",
        "engine_family": "ISX15",
        "description": "High-efficiency lube filter for heavy-duty applications",
        "supplier_name": "Fleetguard",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 22.5,
        "list_price": 38.0,
    },
    {
        "item_code": "BMS0002",
        "item_name": "Fuel Filter Water Separator",
        "engine_family": "ISB6.7",
        "description": "Two-stage fuel filter with water separation",
        "supplier_name": "Fleetguard",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 35.0,
        "list_price": 55.0,
    },
    {
        "item_code": "BMS0003",
        "item_name": "Air Filter Primary",
        "engine_family": "QSB6.7",
        "description": "Primary radial seal air filter",
        "supplier_name": "Donaldson",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 45.0,
        "list_price": 72.0,
    },
    {
        "item_code": "BMS0004",
        "item_name": "Turbocharger Assembly",
        "engine_family": "X15",
        "description": "Genuine Cummins variable geometry turbocharger",
        "supplier_name": "Holset",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 1850.0,
        "list_price": 2450.0,
    },
    {
        "item_code": "BMS0005",
        "item_name": "EGR Valve",
        "engine_family": "ISL9",
        "description": "EGR valve for medium-duty applications",
        "supplier_name": "Cummins Emissions",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 390.0,
        "list_price": 540.0,
    },
    {
        "item_code": "BMS0006",
        "item_name": "Head Gasket Kit",
        "engine_family": "QSK60",
        "description": "Cylinder head gasket set",
        "supplier_name": "Cummins",
        "is_salable": "Y",
        "uom": "KT",
        "standard_cost": 1200.0,
        "list_price": 1520.0,
    },
    {
        "item_code": "BMS0007",
        "item_name": "Piston Kit",
        "engine_family": "QSM11",
        "description": "Forged piston with rings and wrist pin",
        "supplier_name": "Mahle",
        "is_salable": "Y",
        "uom": "KT",
        "standard_cost": 650.0,
        "list_price": 840.0,
    },
    {
        "item_code": "BMS0008",
        "item_name": "Cylinder Liner",
        "engine_family": "ISX15",
        "description": "Wet cylinder liner",
        "supplier_name": "Cummins",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 275.0,
        "list_price": 360.0,
    },
    {
        "item_code": "BMS0009",
        "item_name": "Injector Fuel Pump",
        "engine_family": "ISB6.7",
        "description": "Common rail fuel injector",
        "supplier_name": "Bosch",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 310.0,
        "list_price": 430.0,
    },
    {
        "item_code": "BMS0010",
        "item_name": "Water Pump Assembly",
        "engine_family": "ISX15",
        "description": "Engine coolant water pump",
        "supplier_name": "Cummins Cooling",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 190.0,
        "list_price": 260.0,
    },
    {
        "item_code": "BMS0011",
        "item_name": "Starter Motor",
        "engine_family": "ISL9",
        "description": "12V starter motor",
        "supplier_name": "Delco Remy",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 320.0,
        "list_price": 455.0,
    },
    {
        "item_code": "BMS0012",
        "item_name": "Alternator 150A",
        "engine_family": "ISB6.7",
        "description": "150-amp alternator",
        "supplier_name": "Delco Remy",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 275.0,
        "list_price": 395.0,
    },
    {
        "item_code": "BMS0013",
        "item_name": "Exhaust Manifold",
        "engine_family": "X15",
        "description": "High-temperature exhaust manifold",
        "supplier_name": "Cummins",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 740.0,
        "list_price": 980.0,
    },
    {
        "item_code": "BMS0014",
        "item_name": "Aftertreatment DPF",
        "engine_family": "ISX15",
        "description": "Diesel particulate filter",
        "supplier_name": "Cummins Emissions",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 1450.0,
        "list_price": 1960.0,
    },
    {
        "item_code": "BMS0015",
        "item_name": "Coolant Level Sensor",
        "engine_family": "ISB6.7",
        "description": "Header tank coolant sensor",
        "supplier_name": "Eaton",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 65.0,
        "list_price": 95.0,
    },
    {
        "item_code": "BMS0016",
        "item_name": "Oil Pressure Sensor",
        "engine_family": "QSB6.7",
        "description": "Digital oil pressure sender",
        "supplier_name": "Bosch",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 72.0,
        "list_price": 105.0,
    },
    {
        "item_code": "BMS0017",
        "item_name": "Valve Cover Gasket",
        "engine_family": "ISX15",
        "description": "Molded rubber valve cover gasket",
        "supplier_name": "Cummins",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 48.0,
        "list_price": 72.0,
    },
    {
        "item_code": "BMS0018",
        "item_name": "Timing Actuator",
        "engine_family": "ISL9",
        "description": "Cam timing actuator",
        "supplier_name": "Siemens",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 205.0,
        "list_price": 280.0,
    },
    {
        "item_code": "BMS0019",
        "item_name": "Crankshaft Position Sensor",
        "engine_family": "ISB6.7",
        "description": "Magnetic crank position sensor",
        "supplier_name": "Bosch",
        "is_salable": "Y",
        "uom": "EA",
        "standard_cost": 60.0,
        "list_price": 88.0,
    },
    {
        "item_code": "BMS0020",
        "item_name": "ECM Module",
        "engine_family": "ISX15",
        "description": "Engine control module calibrated for on-highway",
        "supplier_name": "Cummins Electronics",
        "is_salable": "N",
        "uom": "EA",
        "standard_cost": 1800.0,
        "list_price": 2300.0,
    },
]

INVENTORY = []
usa_locations = [
    ("CHI_WH1", "USA", "Chicago"),
    ("DAL_WH1", "USA", "Dallas"),
    ("ATL_WH1", "USA", "Atlanta"),
]
can_locations = [
    ("TOR_WH1", "CAN", "Toronto"),
    ("VAN_WH1", "CAN", "Vancouver"),
    ("MON_WH1", "CAN", "Montreal"),
]
locations = usa_locations + can_locations
for item in ITEMS:
    for code, country, city in locations:
        on_hand = random.randint(0, 180)
        available = max(0, on_hand - random.randint(0, 15))
        INVENTORY.append(
            {
                "item_code": item["item_code"],
                "location_code": code,
                "country": country,
                "city": city,
                "on_hand_qty": on_hand,
                "available_qty": available,
            }
        )

ORDERS = [
    {
        "order_id": "ORD-1001",
        "order_date": "2024-05-12",
        "customer_name": "Northern Fleet Services",
        "ship_to_city": "Toronto",
        "ship_to_country": "Canada",
        "location_code": "TOR_WH1",
        "item_code": "BMS0003",
        "order_qty": 12,
        "status": "Shipped",
    },
    {
        "order_id": "ORD-1002",
        "order_date": "2024-06-03",
        "customer_name": "Midwest Trucking",
        "ship_to_city": "Chicago",
        "ship_to_country": "USA",
        "location_code": "CHI_WH1",
        "item_code": "BMS0001",
        "order_qty": 30,
        "status": "Confirmed",
    },
    {
        "order_id": "ORD-1003",
        "order_date": "2024-06-18",
        "customer_name": "Rocky Mountain Transit",
        "ship_to_city": "Denver",
        "ship_to_country": "USA",
        "location_code": "DAL_WH1",
        "item_code": "BMS0010",
        "order_qty": 6,
        "status": "Delivered",
    },
]

COMPETITOR_SALES = []
brands = ["Cummins", "Caterpillar", "Perkins", "Detroit Diesel", "Volvo Penta"]
item_families = ["ISX", "ISB", "QSK", "Marine"]
start_year = datetime.datetime.now().year - 5
for brand in brands:
    for family in item_families:
        base = random.randint(600, 2200)
        for i in range(5):
            year = start_year + i
            variance = random.randint(-150, 200)
            units = max(120, base + variance + i * random.randint(10, 80))
            revenue = round(units * random.uniform(7500, 11000), 2)
            COMPETITOR_SALES.append(
                {
                    "year": year,
                    "brand": brand,
                    "item_family": family,
                    "units_sold": units,
                    "revenue": revenue,
                }
            )

items_df = pd.DataFrame(ITEMS)
inventory_df = pd.DataFrame(INVENTORY)
orders_df = pd.DataFrame(ORDERS)
competitor_df = pd.DataFrame(COMPETITOR_SALES)


# ----------------------
# Model loading
# ----------------------

def build_model():
    model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        device=-1,
    )


generator = None


def get_generator():
    global generator
    if generator is None:
        generator = build_model()
    return generator


# ----------------------
# Business logic helpers
# ----------------------

def find_item(item_code: str) -> Optional[pd.Series]:
    match = items_df[items_df["item_code"].str.lower() == item_code.lower()]
    if not match.empty:
        return match.iloc[0]
    return None


def get_inventory(item_code: str) -> pd.DataFrame:
    return inventory_df[inventory_df["item_code"].str.lower() == item_code.lower()]


def list_similar_items(partial: str) -> List[str]:
    similar = items_df[items_df["item_code"].str.contains(partial[:4], case=False)]
    return similar["item_code"].tolist()


def forecast_for_series(series: pd.Series) -> Dict[str, float]:
    years = np.arange(len(series))
    coef = np.polyfit(years, series.values, 1)
    next_year = series.values[-1] + coef[0]
    cagr = (series.values[-1] / series.values[0]) ** (1 / max(len(series) - 1, 1)) - 1
    yoy = (series.values[-1] - series.values[-2]) / max(series.values[-2], 1)
    return {"forecast": max(0, next_year), "cagr": cagr, "yoy": yoy}


def build_forecast(
    brand: Optional[str] = None, item_family: Optional[str] = None, item_code: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = competitor_df.copy()
    title_parts = []
    if item_code:
        item = find_item(item_code)
        if item is None:
            raise ValueError("Item code not found in dataset")
        item_family = item["engine_family"]
        brand = "Cummins"
        title_parts.append(f"item {item_code}")
    if brand:
        df = df[df["brand"].str.lower() == brand.lower()]
        title_parts.append(brand)
    if item_family:
        df = df[df["item_family"].str.lower() == item_family.lower()]
        title_parts.append(item_family)
    if df.empty:
        raise ValueError("No historical data available for the requested filter")
    grouped = df.groupby("year")["units_sold"].sum().sort_index()
    stats = forecast_for_series(grouped)
    forecast_year = grouped.index.max() + 1
    forecast_df = pd.DataFrame(
        {
            "year": list(grouped.index) + [forecast_year],
            "units": list(grouped.values) + [round(stats["forecast"], 2)],
            "type": ["historical"] * len(grouped) + ["forecast"],
        }
    )
    stats["label"] = " / ".join(title_parts) if title_parts else "overall"
    return forecast_df, stats


def create_order(record: Dict[str, str]) -> Dict[str, str]:
    global orders_df
    new_id = f"ORD-{random.randint(2000, 9999)}"
    today = datetime.date.today().isoformat()
    order = {
        "order_id": new_id,
        "order_date": today,
        "customer_name": record.get("customer_name", "Walk-in"),
        "ship_to_city": record.get("ship_to_city", ""),
        "ship_to_country": record.get("ship_to_country", ""),
        "location_code": record.get("location_code", ""),
        "item_code": record.get("item_code", ""),
        "order_qty": int(record.get("order_qty", 0)),
        "status": "Created",
    }
    orders_df = pd.concat([orders_df, pd.DataFrame([order])], ignore_index=True)
    return order


def check_order_status(order_id: str) -> Optional[pd.Series]:
    match = orders_df[orders_df["order_id"].str.lower() == order_id.lower()]
    if match.empty:
        return None
    return match.iloc[0]


def competitor_summary() -> pd.DataFrame:
    latest_year = competitor_df["year"].max()
    recent = competitor_df[competitor_df["year"] == latest_year]
    return recent.pivot_table(index="brand", values="units_sold", aggfunc="sum").reset_index()


# ----------------------
# PDF generation
# ----------------------

class ReportBuilder(FPDF):
    def header(self):
        self.set_fill_color(184, 29, 19)
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'BMS AI Assistant Report', ln=True, align='C')
        self.ln(4)
        self.set_text_color(0, 0, 0)


def render_table(pdf: ReportBuilder, title: str, dataframe: pd.DataFrame):
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font('Helvetica', '', 10)
    cols = list(dataframe.columns)
    col_width = 190 / len(cols)
    pdf.set_fill_color(240, 240, 240)
    for col in cols:
        pdf.cell(col_width, 8, str(col), 1, 0, 'C', fill=True)
    pdf.ln(8)
    for _, row in dataframe.iterrows():
        for col in cols:
            pdf.cell(col_width, 8, str(row[col]), 1, 0, 'C')
        pdf.ln(8)
    pdf.ln(4)


def generate_pdf(summary: str, tables: List[Tuple[str, pd.DataFrame]]) -> str:
    pdf = ReportBuilder()
    pdf.add_page()
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(4)
    for title, df in tables:
        render_table(pdf, title, df)
    out_path = os.path.join("/tmp", f"report-{random.randint(1000,9999)}.pdf")
    pdf.output(out_path)
    return out_path


# ----------------------
# Chat orchestration
# ----------------------

def clean_message(text: str) -> str:
    bad_words = ["idiot", "stupid", "dumb"]
    lowered = text.lower()
    if any(w in lowered for w in bad_words):
        return "Let's stay professional. How can I help with inventory, orders, or forecasting?"
    return text


def format_inventory(item_code: str) -> Tuple[str, Optional[pd.DataFrame]]:
    item = find_item(item_code)
    if item is None:
        hints = list_similar_items(item_code)
        suggestion = f" Similar items: {', '.join(hints)}" if hints else ""
        return f"Item {item_code} not found in the current catalog.{suggestion}", None
    stock = get_inventory(item_code)
    summary_lines = [
        f"Item: {item['item_code']} - {item['item_name']}",
        f"Family: {item['engine_family']} | Supplier: {item['supplier_name']} | Salable: {item['is_salable']}",
        f"Pricing: Cost ${item['standard_cost']:.2f}, List ${item['list_price']:.2f}",
        "Availability by location:",
    ]
    table = stock[["location_code", "country", "city", "on_hand_qty", "available_qty"]]
    return "\n".join(summary_lines), table


def respond_with_model(prompt: str, history: List[Dict[str, str]]) -> str:
    if os.getenv("DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        return f"(stubbed response) {prompt}"

    messages = []
    for turn in history[-20:]:
        messages.append(f"User: {turn['user']}")
        messages.append(f"Assistant: {turn['assistant']}")
    messages.append(f"User: {prompt}")
    dialogue = "\n".join(messages)
    system = (
        "You are BMS AI Assistant, a concise and factual helper for Cummins parts. "
        "Keep answers short, avoid inventing data, and prefer bullet points."
    )
    input_text = f"{system}\n\n{dialogue}\nAssistant:"
    result = get_generator()(input_text)[0]["generated_text"]
    return result.split("Assistant:")[-1].strip()


def handle_intent(message: str, state: Dict, history: List[Dict[str, str]]):
    cleaned = clean_message(message)
    lower = cleaned.lower()
    pdf_path = None
    tables_for_report: List[Tuple[str, pd.DataFrame]] = []
    response = None

    if "order status" in lower or lower.startswith("status of"):
        tokens = lower.split()
        order_id = None
        for tok in tokens:
            if tok.startswith("ord-"):
                order_id = tok.upper()
        if not order_id:
            response = "Please share the order ID (e.g., ORD-1002) so I can look it up."
        else:
            order = check_order_status(order_id)
            if order is None:
                response = f"I could not find order {order_id} in the current session."
            else:
                response = (
                    f"Order {order['order_id']} for {order['customer_name']} is currently {order['status']}\n"
                    f"Item {order['item_code']} | Qty {order['order_qty']} | Ship to {order['ship_to_city']}, {order['ship_to_country']}"
                )
    elif "create order" in lower or state.get("order_flow"):
        order_state = state.get("order_flow", {})
        prompts_needed = [
            ("customer_name", "What's the customer name?"),
            ("ship_to_city", "Which city should we ship to?"),
            ("ship_to_country", "Which country?"),
            ("item_code", "What item code is required?"),
            ("order_qty", "How many units are needed?"),
        ]
        for key, question in prompts_needed:
            if key not in order_state and key not in lower:
                response = question
                state["order_flow"] = order_state
                return response, pdf_path, state
        # parse inputs if message not just question
        for key in ["customer_name", "ship_to_city", "ship_to_country", "item_code", "order_qty"]:
            if key in lower and key not in order_state:
                value = cleaned.split(key, 1)[-1].strip().strip(":")
                order_state[key] = value
        if "item_code" not in order_state:
            order_state["item_code"] = cleaned.strip()
        if "order_qty" not in order_state:
            qty_tokens = [int(tok) for tok in lower.split() if tok.isdigit()]
            if qty_tokens:
                order_state["order_qty"] = qty_tokens[0]
        required_missing = [k for k in ["customer_name", "ship_to_city", "ship_to_country", "item_code", "order_qty"] if k not in order_state]
        if required_missing:
            response = f"Missing {', '.join(required_missing)}. Please provide these to create the order."
            state["order_flow"] = order_state
            return response, pdf_path, state
        stock = get_inventory(order_state["item_code"])
        if stock.empty:
            response = f"Item {order_state['item_code']} is not in the catalog."
            state.pop("order_flow", None)
            return response, pdf_path, state
        best_location = stock.sort_values("available_qty", ascending=False).iloc[0]
        order_state.setdefault("location_code", best_location["location_code"])
        order = create_order(order_state)
        state.pop("order_flow", None)
        response = (
            f"Order {order['order_id']} created.\n"
            f"Customer: {order['customer_name']} | Item: {order['item_code']} | Qty: {order['order_qty']}\n"
            f"Ship to: {order['ship_to_city']}, {order['ship_to_country']} from {order['location_code']}"
        )
        tables_for_report.append(("Order", pd.DataFrame([order])))
    elif "inventory" in lower or "stock" in lower or "item details" in lower:
        tokens = [tok for tok in message.replace(",", " ").split() if tok.upper().startswith("BMS")]
        if not tokens:
            response = "Please share the item code (e.g., BMS0001) to check inventory."
        else:
            item_code = tokens[0]
            header, table = format_inventory(item_code)
            response = header
            if table is not None:
                tables_for_report.append((f"Inventory for {item_code}", table))
    elif "forecast" in lower:
        brand = None
        family = None
        code = None
        for tok in message.replace(",", " ").split():
            if tok.upper().startswith("BMS"):
                code = tok
            if tok.lower() in [b.lower() for b in brands]:
                brand = tok
        for fam in item_families:
            if fam.lower() in lower:
                family = fam
        try:
            forecast_df, stats = build_forecast(brand=brand, item_family=family, item_code=code)
            tables_for_report.append(("Forecast", forecast_df))
            response = (
                f"Forecast for {stats['label']}\n"
                f"CAGR: {stats['cagr']*100:.2f}% | YoY: {stats['yoy']*100:.2f}%\n"
                f"Next year projection: {stats['forecast']:.0f} units"
            )
        except ValueError as exc:
            response = str(exc)
    elif "competitor" in lower or "market share" in lower:
        summary = competitor_summary()
        tables_for_report.append(("Market share (units - latest year)", summary))
        response = summary.to_string(index=False)
    elif "pdf" in lower or "report" in lower:
        last_tables = state.get("last_tables")
        last_text = state.get("last_response", "Recent summary from assistant")
        if last_tables:
            pdf_path = generate_pdf(last_text, last_tables)
            response = f"Generated PDF report. You can download it below."
        else:
            response = "There's nothing to report yet. Ask for inventory, orders, or forecasts first."
    else:
        response = respond_with_model(cleaned, history)

    state["last_response"] = response
    if tables_for_report:
        state["last_tables"] = tables_for_report
    if pdf_path is None and ("pdf" in lower or "report" in lower) and state.get("last_tables"):
        pdf_path = generate_pdf(state.get("last_response", ""), state["last_tables"])
    return response, pdf_path, state


def stream_response(message: str, chat_history: List, state: Dict):
    """Stream a response while keeping compatibility with older Gradio Chatbot defaults."""

    state = state or {}
    assistant_history = state.get("history", [])
    response, pdf_path, updated_state = handle_intent(message, state, assistant_history)

    ui_history = list(chat_history or [])
    base_history = ui_history + [(message, "")]

    tokens = response.split()
    buffer = ""
    for tok in tokens:
        buffer += tok + " "
        partial_history = base_history[:-1] + [(message, buffer.strip())]
        yield partial_history, updated_state, pdf_path

    assistant_history.append({"user": message, "assistant": response})
    updated_state["history"] = assistant_history
    yield base_history[:-1] + [(message, response)], updated_state, pdf_path


# ----------------------
# UI definition
# ----------------------

suggestions = [
    "Inventory check for BMS0001 in Canada",
    "Create order for BMS0003 quantity 10 to Toronto",
    "Demand forecast for Cummins ISX family",
    "Compare competitor market share vs Cummins",
    "Generate PDF report for last forecast",
]

css = """
#chatbot .bot {
    background-color: #f7f7f7 !important;
    color: #111;
}
#chatbot .user {
    background-color: #b81d13 !important;
    color: white;
}
#header-title {
    background-color: #b81d13;
    color: white;
    padding: 12px;
    font-weight: bold;
    border-radius: 6px;
    text-align: center;
    font-size: 20px;
}
"""

with gr.Blocks(title="BMS AI Assistant") as demo:
    gr.HTML('<div id="header-title">BMS AI Assistant - Cummins Parts & Service</div>')
    with gr.Row():
        gr.Markdown(
            "**Try asking:**\n- Inventory check for BMS0001 in Canada\n- Create order for BMS0003 quantity 10 to Toronto\n- Demand forecast for Cummins ISX family\n- Compare competitor market share vs Cummins\n- Generate PDF report"
        )
    chatbot = gr.Chatbot(elem_id="chatbot")
    with gr.Row():
        msg = gr.Textbox(label="Ask me about inventory, orders, or forecasts", scale=4)
        submit = gr.Button("Send", variant="primary")
    pdf_download = gr.File(label="PDF Report", interactive=False)
    state = gr.State({})

    submit.click(stream_response, inputs=[msg, chatbot, state], outputs=[chatbot, state, pdf_download])
    msg.submit(stream_response, inputs=[msg, chatbot, state], outputs=[chatbot, state, pdf_download])

if __name__ == "__main__":
    demo.launch(css=css)
