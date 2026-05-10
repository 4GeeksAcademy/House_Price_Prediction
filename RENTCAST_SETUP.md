# RentCast Integration Guide

## Quick Setup (2 minutes)

Your House Price Prediction API is now configured to use **RentCast** for accurate real property data (beds, baths, sq ft, year built, etc.).

### Step 1: Get Your RentCast API Key

1. Visit: https://rentcast.io
2. Sign up (free account available)
3. Copy your **API Key** from the dashboard
4. Note: RentCast offers a free tier + paid options if you need high volume

### Step 2: Enable in `.env`

Edit `.env` and uncomment/fill in your RentCast credentials:

```bash
RENTCAST_API_KEY=your_api_key_here
RENTCAST_API_BASE_URL=https://api.rentcast.io/v1
```

### Step 3: Start the App

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the development app
python -m uvicorn house_price_prediction.app:app --reload

# Or use make
make dev
```

Visit: http://localhost:8000/docs for interactive API docs

---

## How It Works

When you call the predict endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"address": "123 Main St, Seattle, WA 98101"}'
```

The app automatically:

1. **Geocodes** the address → lat/lon (using Nominatim/Census)
2. **Fetches property details** → beds, baths, sq ft, year built (using **RentCast**)
3. **Enriches with Census data** → demographic context
4. **Predicts price** → using trained nationwide model
5. **Returns prediction** → with confidence and feature breakdown

---

## Provider Chain

With RentCast enabled (`PROPERTY_DATA_PROVIDER=free`):

```
1. RentCast API    ← Real property data (beds, baths, sqft, etc.)
2. Census API      ← Fallback: Tract-level demographics
3. Heuristic       ← Fallback: Estimate from address/location
```

### If RentCast Fails

The app seamlessly falls back to Census + Heuristic estimation. This ensures:
- ✅ No broken predictions
- ✅ Graceful degradation
- ✅ Automatic retry on transient errors

---

## Environment Variables

### Required (if using RentCast)

```bash
RENTCAST_API_KEY=your_rentcast_key
```

### Optional

```bash
# Customize provider chain
PROPERTY_DATA_PROVIDER=free          # (default) RentCast → Census → Heuristic
PROPERTY_DATA_PROVIDER=free-fallback # Multi-source chain

# Customize timeouts
PROVIDER_TIMEOUT_SECONDS=25.0
PROVIDER_MAX_RETRIES=1

# Optional: Add walkability scores
WALKSCORE_API_KEY=your_walkscore_key
```

---

## Production Deployment

### On Render.com (via render.yaml)

The app is already configured for Render deployment:

```yaml
env:
  - key: PROPERTY_DATA_PROVIDER
    value: free
  
  - key: RENTCAST_API_KEY
    value: your_rentcast_key_here  # <-- Set this
```

### On Docker/AWS/Other Platforms

Set these environment variables:

```bash
RENTCAST_API_KEY=your_key
PROPERTY_DATA_PROVIDER=free
MODEL_PATH=models/nationwide_smart_router.joblib
```

---

## Testing the Integration

```bash
# Activate venv
source .venv/bin/activate

# Test RentCast with a real address
python examples/test_rentcast_lookup.py "1234 5th Ave, Seattle, WA 98104"

# Or integrate in your app
from house_price_prediction.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.post("/predict", json={
    "address": "1234 5th Ave, Seattle, WA 98104"
})
print(response.json())
```

---

## Troubleshooting

### "Missing RENTCAST_API_KEY"
- Check `.env` has `RENTCAST_API_KEY=your_key`
- Verify key is valid at https://rentcast.io/dashboard

### "RentCast lookup failed"
- Check internet connection
- Verify address format (should include street, city, state, zip)
- The app will automatically fall back to Census data

### "Slow predictions"
- RentCast calls + Census calls can take 3-5 seconds total
- This is normal. Caching (`PROVIDER_RESPONSE_CACHE_MAX_AGE_HOURS=24`) prevents repeated lookups

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ HTTP Request: POST /predict                         │
│ Body: { "address": "..." }                          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Geocoding            │
        │ (Nominatim/Census)   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │ Property Data (RentCast)      │
        │ • Beds/Baths/SqFt            │
        │ • Year Built/Lot Size         │
        │ • Property Type               │
        └──────────┬────────────────────┘
                   │
          ┌────────┴──────────┐
          │                   │
          ▼                   ▼
    ┌──────────────┐  ┌──────────────┐
    │ Census Data  │  │ Heuristic    │
    │ (if needed)  │  │ (if needed)  │
    └──────────────┘  └──────────────┘
          │                   │
          └────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Price Prediction     │
        │ (SmartRouter Model)  │
        └──────────┬───────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │ HTTP Response: Price + Details  │
    │ • Prediction ± confidence       │
    │ • Feature breakdown             │
    │ • Data sources used             │
    └─────────────────────────────────┘
```

---

## Next Steps

1. ✅ Add your RentCast API key to `.env`
2. ✅ Start the app (`make dev` or `uvicorn`)
3. ✅ Test via `/docs` or `curl`
4. ✅ Deploy to production (Render/Docker/etc)
5. ✅ Monitor predictions in the database

---

## Support

- **RentCast API Docs**: https://www.rentcast.io/docs
- **This App Docs**: http://localhost:8000/docs
- **Source Code**: See `src/house_price_prediction/infrastructure/providers/`
