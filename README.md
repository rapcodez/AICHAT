---
title: BMS AI Assistant (Cummins Parts Chatbot)
emoji: 🚚
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.41.0"
app_file: app.py
pinned: false
---

# BMS AI Assistant (Cummins Parts Chatbot)

An open-source Gradio chatbot for Cummins engines and spare parts. It runs fully on free Hugging Face Spaces CPU tiers and uses only open models and Python libraries.

## Features
- Natural chat using the `Qwen/Qwen2-0.5B-Instruct` open-source LLM.
- Dummy but realistic datasets for parts, inventory across USA/Canada, orders, and competitor sales.
- Business tools:
  - Demand forecasting with simple trends (CAGR + YoY) per brand, family, or item.
  - Inventory lookup with item details and per-location availability.
  - Order creation workflow with validation and order status lookups.
  - Competitor analysis (market share by latest year) and forecast support.
  - PDF report generation for inventory, forecasts, orders, or comparisons.
- Session memory for the last 20 turns and simple profanity handling.

## Project Structure
- `app.py` – Main Gradio application, data seeding, business logic, PDF generation, and chat orchestration.
- `requirements.txt` – Python dependencies for Hugging Face Spaces.

## Running Locally
1. Create and activate a virtual environment (optional).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Gradio app:
   ```bash
   python app.py
   ```
4. Open the printed local URL in your browser to chat.

## Deploying to Hugging Face Spaces
1. Create a new Space (Gradio + CPU is sufficient).
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. (Optional) Set `LLM_MODEL` env var in the Space settings to swap in any compatible instruct model hosted on Hugging Face.
4. The Space will build and launch automatically; open the public URL to start chatting.

## Example Prompts
- "Inventory check for BMS0001 in Canada"
- "Create order for BMS0003 quantity 10 to Toronto"
- "Demand forecast for Cummins ISX family"
- "Compare competitor market share vs Cummins"
- "Generate PDF report for last forecast"
