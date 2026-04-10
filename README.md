# APEX — Dynamic Pricing Engine

Real-time dynamic pricing, personalized recommendations, and A/B testing engine for e-commerce.

## Features
- **Dynamic Pricing**: `P = base × (1 + min(15%, clicks/10))` with scarcity & competitor adjustments
- **Fairness Guardrails**: Hard +15% price ceiling, cost-floor protection
- **Session-Based Recommendations**: Category affinity + trending products
- **A/B Testing**: Static vs Dynamic pricing with conversion tracking
- **Price Rationale**: Every price change is explained
- **Sub-1ms Latency**: In-memory Python dictionary store

## Run Locally
```bash
pip install -r requirements.txt
python server.py
# Open http://localhost:8000
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python + FastAPI |
| Frontend | React (CDN) + TailwindCSS |
| Data Store | In-memory Python dict |
| Data | Parquet files (5K+ products) |
