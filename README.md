# Mai Shun Yun Inventory Intelligence Dashboard

An interactive Streamlit app that helps restaurant managers monitor ingredient health, forecast demand, and act on data-driven reorder recommendations. The project unifies multiple data sources provided in the MSY challenge and layers on AI-assisted guidance via Google Gemini plus optional MongoDB persistence for saved alerts.

## Key Features
- **Unified data model**: Load monthly sales summaries, recipe usage, and shipment schedules from the `MSYData/` folder.
- **Interactive visuals**: Explore revenue trends, sales category mix, and ingredient days-on-hand in a single dashboard.
- **Predictive analytics**: Generate Holt-Winters or moving-average forecasts to anticipate stock depletion and suggest reorder quantities.
- **AI assistant (Gemini Track)**: Ask natural-language questions such as “Which items should I reorder this week?” and receive contextual answers grounded in your metrics.
- **Alert persistence (MongoDB Track)**: Save high-priority reorder notes to MongoDB so managers can track actions across shifts.

## Project Structure
```
├── data_loader.py        # Centralised loaders for sales, recipe, and shipment data
├── features.py           # Feature engineering + insight generation logic
├── forecast.py           # Time-series helpers (moving average, Holt-Winters)
├── simulate.py           # Scenario planning utilities for what-if analysis
├── dashboard.py          # Streamlit app (UI, Gemini integration, Mongo alerts)
├── mongo_utils.py        # Lightweight MongoDB helper functions
├── requirements.txt      # Reproducible dependency list
├── visualize.py          # (Partner-owned) standalone Matplotlib prototype
└── MSYData/              # Provided challenge datasets and assets
```

## Getting Started

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Optional Integrations
- **Gemini Assistant**
  ```bash
  export GOOGLE_API_KEY="your-google-genai-key"
  export GEMINI_MODEL="gemini-1.5-flash"   # Optional override
  ```
- **MongoDB Alerts**
  ```bash
  export MONGO_URI="your-mongodb-connection-string"
  export MONGO_DB="inventory_dashboard"          # optional
  export MONGO_COLLECTION="alerts"               # optional
  ```
  If `MONGO_URI` is not set, the dashboard still runs; alert persistence simply stays disabled.

### 3. Launch the Dashboard
```bash
streamlit run dashboard.py
```

The app automatically discovers the datasets inside `MSYData/`. Use the sidebar filters and chat assistant to explore insights.

## Data Integration Notes
- **Sales Summaries**: Monthly Excel files such as `June_Data_Matrix.xlsx` are parsed into tidy tables (period, category, count, revenue).
- **Recipe Reference**: Ingredient usage per menu item (`MSY Data - Ingredient.csv`) is normalized and combined with sales to estimate ingredient consumption.
- **Shipments**: Recurring delivery schedules (`MSY Data - Shipment.csv`) help approximate inflow vs. usage to compute days-on-hand heuristics.
- **Extensibility**: If your data cleaning teammate produces more granular sales exports (e.g., item-level POS data), drop them into `MSYData/` and update the loader to ingest the new schema. Feature functions will automatically prefer explicit `item_name` columns when present.

## Example Insights
- **Stock-at-risk list**: Ingredients sorted by projected depletion date plus suggested reorder quantity (based on forecasted demand).
- **Seasonality call-outs**: Comparing revenue and transaction deltas month over month to spot shoulder-season dips.
- **AI-driven Q&A**: Ask “What ingredient usage spikes should I prepare for next month?” and Gemini summarises metrics with actionable recommendations.
- **Cross-shift collaboration**: Save alerts (e.g., “Expedite bok choy delivery”) to MongoDB, so the next manager sees the latest status.

## Sponsor & Prize Tracks
- **Google Gemini**: Integrated via `google-generativeai` to deliver natural-language coaching grounded in live dashboard metrics.
- **MongoDB Atlas**: Optional persistence layer for alerts, enabling teams to log decisions and follow-ups directly from the dashboard.

Potential future extensions include deploying the app on Vultr for always-on hosting, piping telemetry to Snowflake for enterprise analytics, or generating daily voice briefings with ElevenLabs.

## Demo Guidance
Record a short walkthrough highlighting:
1. Dashboard navigation (metrics, charts, ingredient table).
2. An example scenario (e.g., identify low-stock ingredients and save an alert).
3. Gemini assistant responding to a reorder question.
4. Optional: Show MongoDB entries appearing after a saved alert.

## Development Tips
- Keep heavy data cleaning logic in separate scripts/notebooks; export ready-to-use tables to `MSYData/`.
- The feature layer is unit-test friendly—feed custom DataFrames to validate calculations without running Streamlit.
- `simulate.py` offers starter utilities for menu-launch or demand-shock what-if analyses; wire them into the dashboard as stretch goals.

---

Happy hacking! 🍜
