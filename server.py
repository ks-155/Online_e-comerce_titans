"""
============================================================
  APEX — Dynamic Pricing & Recommendation Engine
  FastAPI Backend · Real-Time · Sub-200ms Latency
============================================================

  Run:     python server.py
  Open:    http://localhost:8000

  Data:    Loads from parquet + CSV files
  Pricing: Demand-responsive + Scarcity + Competitor-aware
  Recs:    Session-based category affinity
  A/B:     Control vs Dynamic pricing groups
============================================================
"""

import os
import sys
import time
import random
import csv
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(BASE_DIR, "Problem Statement 3 Sample Data")
MAX_PRICE_INCREASE = 0.15   # 15% hard ceiling (fairness)
MAX_PRICE_DECREASE = 0.10   # 10% max discount
DEMAND_DIVISOR = 10         # clicks / 10 = demand factor

# ─────────────────────────────────────────────
# Data Loading — In-Memory Store (fast!)
# ─────────────────────────────────────────────
PRODUCTS = {}
COMPETITOR_DATA = {}
USER_SEGMENTS = {}
CATEGORIES = defaultdict(list)   # category -> [sku_ids]

def load_data():
    """Load real data from parquet files + CSV into memory."""
    global PRODUCTS, COMPETITOR_DATA, USER_SEGMENTS, CATEGORIES

    # --- 1. Load Product Catalog from parquet ---
    try:
        import pandas as pd
        catalog = pd.read_parquet(os.path.join(PARQUET_DIR, "product_catalog.parquet"))
        for _, row in catalog.head(50).iterrows():  # Top 50 products for demo speed
            sku = row["sku_id"]
            PRODUCTS[sku] = {
                "sku_id": sku,
                "name": row["product_name"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "brand": row["brand"],
                "base_price": round(float(row["base_price_usd"]), 2),
                "cost_price": round(float(row["cost_price_usd"]), 2),
                "current_price": round(float(row["current_price_usd"]), 2),
                "min_price": round(float(row["min_price_usd"]), 2),
                "max_price": round(float(row["max_price_usd"]), 2),
                "inventory": int(row["inventory_count"]),
                "rating": round(float(row["avg_rating"]), 1),
                "review_count": int(row["review_count"]),
                "tags": row.get("tags", "[]"),
                # Live tracking fields
                "views": 0,
                "clicks": 0,
                "add_to_carts": 0,
                "purchases": 0,
                "price_history": [round(float(row["current_price_usd"]), 2)],
            }
            CATEGORIES[row["category"]].append(sku)
        print(f"  [OK] Loaded {len(PRODUCTS)} products from parquet")
    except Exception as e:
        print(f"  [WARN] Parquet load failed ({e}), falling back to CSV...")
        _load_from_csv()

    # --- 2. Load Competitor Pricing (sample) ---
    try:
        import pandas as pd
        comp = pd.read_parquet(os.path.join(PARQUET_DIR, "competitor_pricing_feed.parquet"))
        # Keep latest competitor price per SKU (just top products)
        for sku in list(PRODUCTS.keys()):
            sku_data = comp[comp["sku_id"] == sku]
            if len(sku_data) > 0:
                latest = sku_data.iloc[-1]
                COMPETITOR_DATA[sku] = {
                    "competitor": latest["competitor"],
                    "competitor_price": round(float(latest["competitor_price"]), 2),
                    "is_on_promotion": bool(latest["is_on_promotion"]),
                }
        print(f"  [OK] Loaded competitor data for {len(COMPETITOR_DATA)} products")
    except Exception as e:
        print(f"  [WARN] Competitor data load skipped: {e}")

    # --- 3. Load User Segments (sample) ---
    try:
        import pandas as pd
        users = pd.read_parquet(os.path.join(PARQUET_DIR, "user_segment_profiles.parquet"))
        for _, row in users.head(200).iterrows():
            uid = row["user_id"]
            USER_SEGMENTS[uid] = {
                "segment": row["segment"],
                "willingness_to_pay": round(float(row["willingness_to_pay_multiplier"]), 3),
                "preferred_categories": row.get("preferred_categories", "[]"),
                "device": row.get("device_type", "desktop"),
            }
        print(f"  [OK] Loaded {len(USER_SEGMENTS)} user profiles")
    except Exception as e:
        print(f"  [WARN] User segments load skipped: {e}")


def _load_from_csv():
    """Fallback: load from products.csv on Desktop."""
    csv_path = os.path.join(BASE_DIR, "products.csv")
    if not os.path.exists(csv_path):
        print("  [WARN] No data files found, using demo products")
        _load_demo_products()
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 50:
                break
            sku = row["product_id"]
            base = float(row["price"])
            PRODUCTS[sku] = {
                "sku_id": sku,
                "name": row["product_name"],
                "category": row["category"],
                "subcategory": row.get("subcategory", ""),
                "brand": row.get("brand", ""),
                "base_price": round(base, 2),
                "cost_price": round(float(row.get("cost_price", base * 0.5)), 2),
                "current_price": round(base, 2),
                "min_price": round(base * 0.70, 2),
                "max_price": round(base * 1.30, 2),
                "inventory": int(row.get("inventory", 100)),
                "rating": round(float(row.get("rating", 3.5)), 1),
                "review_count": int(row.get("review_count", 100)),
                "tags": "[]",
                "views": 0,
                "clicks": 0,
                "add_to_carts": 0,
                "purchases": 0,
                "price_history": [round(base, 2)],
            }
            CATEGORIES[row["category"]].append(sku)
    print(f"  [OK] Loaded {len(PRODUCTS)} products from CSV")


def _load_demo_products():
    """Emergency fallback with hardcoded demo products."""
    demo = [
        ("SKU001", "Wireless Headphones", "Electronics", "Audio", "SoundMax", 129.99),
        ("SKU002", "Mechanical Keyboard", "Electronics", "Peripherals", "KeyTech", 89.99),
        ("SKU003", "Smart Watch Pro", "Electronics", "Wearables", "TechWear", 299.99),
        ("SKU004", "Running Shoes Elite", "Fashion", "Shoes", "RunFast", 159.99),
        ("SKU005", "Organic Face Cream", "Beauty", "Skincare", "GlowUp", 45.99),
        ("SKU006", "Yoga Mat Premium", "Sports", "Fitness", "ZenFit", 69.99),
        ("SKU007", "LED Desk Lamp", "Home & Kitchen", "Lighting", "BrightLife", 39.99),
        ("SKU008", "Python Programming Book", "Books", "Education", "LearnCo", 34.99),
    ]
    for sku, name, cat, sub, brand, price in demo:
        PRODUCTS[sku] = {
            "sku_id": sku, "name": name, "category": cat, "subcategory": sub,
            "brand": brand, "base_price": price, "cost_price": round(price * 0.5, 2),
            "current_price": price, "min_price": round(price * 0.70, 2),
            "max_price": round(price * 1.30, 2), "inventory": 100,
            "rating": 4.0, "review_count": 500, "tags": "[]",
            "views": 0, "clicks": 0, "add_to_carts": 0, "purchases": 0,
            "price_history": [price],
        }
        CATEGORIES[cat].append(sku)


# ─────────────────────────────────────────────
# In-Memory Session & Event Tracking
# ─────────────────────────────────────────────
SESSION_STORE = defaultdict(lambda: {
    "clicks": [],          # list of sku_ids clicked
    "categories": [],      # categories browsed
    "total_events": 0,
    "start_time": time.time(),
    "ab_group": None,
})

EVENT_LOG = []             # Global event log
ANALYTICS = {
    "total_events": 0,
    "total_revenue_static": 0.0,
    "total_revenue_dynamic": 0.0,
    "group_a_conversions": 0,
    "group_b_conversions": 0,
    "group_a_sessions": 0,
    "group_b_sessions": 0,
}


# ─────────────────────────────────────────────
# 💰 Dynamic Pricing Engine
# ─────────────────────────────────────────────
def compute_dynamic_price(product: dict) -> tuple:
    """
    Compute dynamic price with full rationale.
    
    Formula: P = base_price × (1 + min(0.15, clicks/10))
    Adjustments: scarcity, competitor, demand velocity
    
    Returns: (new_price, rationale_list)
    """
    base = product["base_price"]
    clicks = product["clicks"]
    inventory = product["inventory"]
    sku = product["sku_id"]
    rationale = []

    # ─── 1. Demand-Responsive Component ───
    # P = base × (1 + min(0.15, clicks/10))
    demand_factor = min(MAX_PRICE_INCREASE, clicks / DEMAND_DIVISOR)
    demand_multiplier = 1 + demand_factor
    if demand_factor > 0.01:
        rationale.append(f"📈 High demand: +{demand_factor*100:.1f}% ({clicks} clicks)")
    
    price = base * demand_multiplier

    # ─── 2. Scarcity Adjustment ───
    if inventory < 20:
        scarcity_boost = min(0.05, (20 - inventory) / 200)  # up to +5%
        price *= (1 + scarcity_boost)
        rationale.append(f"🔥 Low stock ({inventory} left): +{scarcity_boost*100:.1f}%")
    elif inventory > 400:
        surplus_discount = min(0.05, (inventory - 400) / 2000)  # up to -5%
        price *= (1 - surplus_discount)
        rationale.append(f"📦 High stock: -{surplus_discount*100:.1f}% to move inventory")

    # ─── 3. Competitor Price Check ───
    if sku in COMPETITOR_DATA:
        comp = COMPETITOR_DATA[sku]
        comp_price = comp["competitor_price"]
        if comp_price < price * 0.95:  # Competitor is >5% cheaper
            adjustment = min(0.03, (price - comp_price) / price * 0.5)
            price *= (1 - adjustment)
            rationale.append(f"🏪 Competitor match ({comp['competitor']}): -{adjustment*100:.1f}%")
        elif comp["is_on_promotion"]:
            rationale.append(f"⚡ Competitor {comp['competitor']} running promotion")

    # ─── 4. Enforce Fairness Guardrails ───
    # Hard ceiling: never more than +15% above base
    ceiling = base * (1 + MAX_PRICE_INCREASE)
    floor = max(product["cost_price"] * 1.05, base * (1 - MAX_PRICE_DECREASE))
    
    if price > ceiling:
        price = ceiling
        rationale.append(f"🛡️ Price ceiling applied (max +{MAX_PRICE_INCREASE*100:.0f}%)")
    if price < floor:
        price = floor
        rationale.append(f"🛡️ Price floor applied (margin protection)")

    # Clamp to min/max from catalog
    price = max(product["min_price"], min(price, product["max_price"]))
    
    price = round(price, 2)
    
    if not rationale:
        rationale.append("📊 Base price — no demand signals yet")

    return price, rationale


# ─────────────────────────────────────────────
# 🧠 Recommendation Engine
# ─────────────────────────────────────────────
def get_recommendations(session: dict, current_sku: str = None, limit: int = 5) -> list:
    """
    Session-based recommendations:
    1. Same category as recent clicks
    2. Trending products (most viewed)
    3. Exclude already-clicked items
    """
    clicked = set(session.get("clicks", []))
    recent_categories = session.get("categories", [])[-5:]  # last 5 categories

    candidates = []

    # Step 1: Category-based (what user is browsing)
    for cat in recent_categories:
        for sku in CATEGORIES.get(cat, []):
            if sku not in clicked and sku in PRODUCTS:
                p = PRODUCTS[sku]
                candidates.append({
                    "sku_id": sku,
                    "name": p["name"],
                    "category": p["category"],
                    "price": p["current_price"],
                    "rating": p["rating"],
                    "reason": f"Because you browsed {cat}",
                    "score": p["views"] + p["rating"] * 10,
                })

    # Step 2: Fill with trending if not enough
    if len(candidates) < limit:
        trending = sorted(PRODUCTS.values(), key=lambda x: x["views"], reverse=True)
        for p in trending:
            if p["sku_id"] not in clicked and not any(c["sku_id"] == p["sku_id"] for c in candidates):
                candidates.append({
                    "sku_id": p["sku_id"],
                    "name": p["name"],
                    "category": p["category"],
                    "price": p["current_price"],
                    "rating": p["rating"],
                    "reason": "Trending now",
                    "score": p["views"] + p["rating"] * 5,
                })
            if len(candidates) >= limit * 2:
                break

    # Sort by relevance score, return top N
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    load_data()
    yield

app = FastAPI(
    title="APEX Dynamic Pricing Engine",
    description="Real-time dynamic pricing, recommendations & A/B testing",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve Frontend ──
@app.get("/")
def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "APEX API running. Place index.html in same folder."}


# ── GET /products ──
@app.get("/products")
def list_products(limit: int = Query(20, description="Max products to return")):
    """Return all products with current dynamic prices."""
    products = list(PRODUCTS.values())[:limit]
    return {
        "products": [
            {
                "sku_id": p["sku_id"],
                "name": p["name"],
                "category": p["category"],
                "subcategory": p["subcategory"],
                "brand": p["brand"],
                "base_price": p["base_price"],
                "current_price": p["current_price"],
                "inventory": p["inventory"],
                "rating": p["rating"],
                "review_count": p["review_count"],
                "views": p["views"],
                "clicks": p["clicks"],
            }
            for p in products
        ],
        "total": len(PRODUCTS),
    }


# ── POST /event ──
@app.post("/event")
def record_event(
    sku_id: str = Query(..., description="Product SKU"),
    event_type: str = Query("click", description="click|view|add_to_cart|purchase"),
    user_id: str = Query("U001", description="User ID"),
    session_id: str = Query("S001", description="Session ID"),
):
    """
    Record a user event and trigger price recalculation.
    This is the core endpoint for the real-time pipeline.
    """
    start = time.time()

    if sku_id not in PRODUCTS:
        return {"error": f"Product {sku_id} not found"}

    product = PRODUCTS[sku_id]
    session = SESSION_STORE[session_id]
    old_price = product["current_price"]

    # ─ Track event ─
    product["views"] += 1
    if event_type == "click":
        product["clicks"] += 1
    elif event_type == "add_to_cart":
        product["add_to_carts"] += 1
        product["clicks"] += 1  # cart = strong signal
    elif event_type == "purchase":
        product["purchases"] += 1
        product["clicks"] += 2  # purchase = strongest signal
        product["inventory"] = max(0, product["inventory"] - 1)

    # ─ Update session ─
    session["clicks"].append(sku_id)
    session["categories"].append(product["category"])
    session["total_events"] += 1

    # ─ A/B Group Assignment ─
    if session["ab_group"] is None:
        session["ab_group"] = "B" if hash(session_id) % 2 == 0 else "A"

    # ─ Compute new price ─
    new_price, rationale = compute_dynamic_price(product)

    ab_group = session["ab_group"]
    if ab_group == "A":
        # Control group: static pricing
        display_price = product["base_price"]
        rationale = ["📊 Control group: static base price"]
        ANALYTICS["group_a_sessions"] += 1
        if event_type == "purchase":
            ANALYTICS["group_a_conversions"] += 1
            ANALYTICS["total_revenue_static"] += product["base_price"]
    else:
        # Treatment group: dynamic pricing
        display_price = new_price
        product["current_price"] = new_price
        product["price_history"].append(new_price)
        ANALYTICS["group_b_sessions"] += 1
        if event_type == "purchase":
            ANALYTICS["group_b_conversions"] += 1
            ANALYTICS["total_revenue_dynamic"] += new_price

    # ─ Get recommendations ─
    recs = get_recommendations(session, current_sku=sku_id)

    # ─ Log event ─
    ANALYTICS["total_events"] += 1
    latency_ms = round((time.time() - start) * 1000, 1)

    EVENT_LOG.append({
        "event_type": event_type,
        "sku_id": sku_id,
        "user_id": user_id,
        "session_id": session_id,
        "old_price": old_price,
        "new_price": display_price,
        "ab_group": ab_group,
        "latency_ms": latency_ms,
        "timestamp": time.time(),
    })

    return {
        "status": "ok",
        "sku_id": sku_id,
        "product_name": product["name"],
        "category": product["category"],
        "old_price": old_price,
        "new_price": display_price,
        "price_change": round(display_price - old_price, 2),
        "price_change_pct": round((display_price - old_price) / old_price * 100, 2) if old_price > 0 else 0,
        "direction": "up" if display_price > old_price else ("down" if display_price < old_price else "same"),
        "rationale": rationale,
        "ab_group": ab_group,
        "inventory": product["inventory"],
        "demand_clicks": product["clicks"],
        "recommendations": recs,
        "latency_ms": latency_ms,
    }


# ── GET /price ──
@app.get("/price")
def get_price(product_id: str = Query(..., description="Product SKU ID")):
    """Get current dynamic price for a product (backward compat)."""
    # Support both "SKU001000" and simple numeric IDs
    sku = product_id if product_id in PRODUCTS else None
    if not sku:
        # Try matching by index
        skus = list(PRODUCTS.keys())
        try:
            idx = int(product_id) - 1
            if 0 <= idx < len(skus):
                sku = skus[idx]
        except (ValueError, IndexError):
            pass

    if not sku or sku not in PRODUCTS:
        return {"error": f"Product {product_id} not found", "price": 0}

    product = PRODUCTS[sku]
    product["views"] += 1
    product["clicks"] += 1

    new_price, rationale = compute_dynamic_price(product)
    old_price = product["current_price"]
    product["current_price"] = new_price

    return {
        "product_id": sku,
        "name": product["name"],
        "price": new_price,
        "old_price": old_price,
        "base_price": product["base_price"],
        "direction": "up" if new_price > old_price else "down",
        "rationale": rationale,
        "inventory": product["inventory"],
        "views": product["views"],
        "clicks": product["clicks"],
    }


# ── GET /recommendations ──
@app.get("/recommendations")
def get_recs(
    user_id: str = Query("U001"),
    session_id: str = Query("S001"),
):
    """Get personalized recommendations for a session."""
    session = SESSION_STORE[session_id]
    recs = get_recommendations(session, limit=5)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "recommendations": recs,
    }


# ── GET /analytics ──
@app.get("/analytics")
def analytics():
    """Dashboard analytics: A/B test results, revenue, engagement."""
    conv_a = (ANALYTICS["group_a_conversions"] / max(1, ANALYTICS["group_a_sessions"])) * 100
    conv_b = (ANALYTICS["group_b_conversions"] / max(1, ANALYTICS["group_b_sessions"])) * 100
    lift = round(conv_b - conv_a, 2) if conv_a > 0 else 0

    top_products = sorted(
        PRODUCTS.values(),
        key=lambda x: x["views"],
        reverse=True
    )[:5]

    recent_events = EVENT_LOG[-10:]  # Last 10 events

    return {
        "total_events": ANALYTICS["total_events"],
        "products_tracked": len(PRODUCTS),
        "avg_latency_ms": round(
            sum(e["latency_ms"] for e in EVENT_LOG[-50:]) / max(1, len(EVENT_LOG[-50:])),
            1
        ) if EVENT_LOG else 0,
        "ab_test": {
            "group_a_static": {
                "sessions": ANALYTICS["group_a_sessions"],
                "conversions": ANALYTICS["group_a_conversions"],
                "conversion_rate": round(conv_a, 2),
                "revenue": round(ANALYTICS["total_revenue_static"], 2),
            },
            "group_b_dynamic": {
                "sessions": ANALYTICS["group_b_sessions"],
                "conversions": ANALYTICS["group_b_conversions"],
                "conversion_rate": round(conv_b, 2),
                "revenue": round(ANALYTICS["total_revenue_dynamic"], 2),
            },
            "lift_pct": lift,
            "summary": f"Dynamic pricing {'increased' if lift > 0 else 'decreased'} conversion by {abs(lift):.1f}%",
        },
        "top_products": [
            {"name": p["name"], "views": p["views"], "clicks": p["clicks"], "price": p["current_price"]}
            for p in top_products
        ],
        "recent_events": recent_events,
    }


# ── GET /health ──
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "products_loaded": len(PRODUCTS),
        "competitors_loaded": len(COMPETITOR_DATA),
        "user_segments_loaded": len(USER_SEGMENTS),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


_start_time = time.time()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  APEX -- Dynamic Pricing Engine v2.0")
    print("=" * 55)
    print(f"  Data dir : {PARQUET_DIR}")
    print(f"  API      : http://localhost:8000")
    print(f"  Frontend : http://localhost:8000")
    print("=" * 55)
    print()
    port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)

