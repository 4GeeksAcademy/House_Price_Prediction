"""
Test the trained model against real addresses with realistic feature values.
"""
import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/house_price_model.pkl")

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)


# CensusMedianValue = ZIP-level median price from the training CSV (data/raw/Housing.csv)
# PricePerSqft     = Zillow-listed price / square footage (actual market signal)
# LandValue        = estimated as 25% of Zillow-listed price

properties = [
    {
        "_address": "123 Maple St, Little Rock, AR 72204",
        "_label": "Starter home",
        "_zillow_price": 154000,   # estimated market value for this ZIP
        "LotArea": 8750.0,
        "OverallQual": 6.0,
        "OverallCond": 5.0,
        "YearBuilt": 1998.0,
        "YearRemodAdd": 2010.0,
        "GrLivArea": 1620.0,
        "FullBath": 2.0,
        "HalfBath": 0.0,
        "BedroomAbvGr": 3.0,
        "TotRmsAbvGrd": 7.0,
        "Fireplaces": 1.0,
        "GarageCars": 2.0,
        "GarageArea": 440.0,
        "NeighborhoodScore": 65.0,
        "CensusMedianValue": 134900.0,   # ZIP 72204 median from CSV
        "MedianIncomeK": 52.0,
        "OwnerOccupiedRate": 0.62,
        "SchoolDistrictRating": 6.0,
        "WalkScore": 55.0,
        "HOAFee": 0.0,
        "PricePerSqft": 95.0,            # ZIP 72204 mean $/sqft from CSV
        "LandValue": 34000.0,
        "PropertyType": "single_family",
        "City": "Little Rock",
        "ZipCode": "72204",
        "State": "AR",
    },
    {
        "_address": "4500 Cahaba River Blvd, Birmingham, AL 35244",
        "_label": "Suburban family home",
        "_zillow_price": 408700,         # Zillow Zestimate
        "LotArea": 7423.0,               # Zillow: 7,423 sq ft lot
        "OverallQual": 7.0,
        "OverallCond": 7.0,
        "YearBuilt": 2012.0,             # Zillow: Built in 2012
        "YearRemodAdd": 2012.0,
        "GrLivArea": 1950.0,             # Zillow: 1,950 sqft
        "FullBath": 2.0,                 # Zillow: 2 baths
        "HalfBath": 0.0,
        "BedroomAbvGr": 3.0,             # Zillow: 3 beds
        "TotRmsAbvGrd": 7.0,
        "Fireplaces": 1.0,
        "GarageCars": 2.0,
        "GarageArea": 440.0,
        "NeighborhoodScore": 78.0,
        "CensusMedianValue": 446950.0,   # ZIP 35244 median from training CSV
        "MedianIncomeK": 85.0,
        "OwnerOccupiedRate": 0.76,
        "SchoolDistrictRating": 9.0,
        "WalkScore": 42.0,
        "HOAFee": 100.0,
        "PricePerSqft": 209.6,           # Zillow: $210/sqft (408700 / 1950)
        "LandValue": 102175.0,           # 408700 * 0.25
        "PropertyType": "single_family",
        "City": "Birmingham",
        "ZipCode": "35244",              # Zillow: AL 35244
        "State": "AL",
    },
    {
        "_address": "1931 Deco Dr, Kennesaw, GA 30144",
        "_label": "Suburban townhome",
        "_zillow_price": 449900,             # Zillow list price
        "LotArea": 2000.0,                   # small private lot typical for townhome
        "OverallQual": 8.0,
        "OverallCond": 8.0,
        "YearBuilt": 2021.0,                 # newer Kennesaw development
        "YearRemodAdd": 2021.0,
        "GrLivArea": 1824.0,                 # Zillow: 1,824 sqft
        "FullBath": 3.0,                     # Zillow: 3.5 ba → 3 full
        "HalfBath": 1.0,                     # Zillow: 3.5 ba → 1 half
        "BedroomAbvGr": 3.0,                 # Zillow: 3 beds
        "TotRmsAbvGrd": 8.0,
        "Fireplaces": 1.0,
        "GarageCars": 2.0,
        "GarageArea": 400.0,
        "NeighborhoodScore": 75.0,
        "CensusMedianValue": 380000.0,       # ZIP 30144 Kennesaw GA median
        "MedianIncomeK": 85.0,
        "OwnerOccupiedRate": 0.65,
        "SchoolDistrictRating": 8.0,         # Cobb County schools
        "WalkScore": 45.0,
        "HOAFee": 250.0,                     # typical Kennesaw townhome HOA
        "PricePerSqft": 246.7,              # 449900 / 1824
        "LandValue": 67485.0,               # 449900 × 0.15 (townhome land share)
        "PropertyType": "townhouse",
        "City": "Kennesaw",
        "ZipCode": "30144",
        "State": "GA",
    },
    {
        "_address": "5270 Lodewyck St, Detroit, MI 48224",
        "_label": "Distressed starter home",
        "_zillow_price": 53000,              # Zillow list price ($-- Zestimate, $61/sqft)
        "LotArea": 4792.0,                   # Zillow: 4,791.6 sq ft lot
        "OverallQual": 4.0,
        "OverallCond": 5.0,
        "YearBuilt": 1930.0,                 # Zillow: Built in 1930
        "YearRemodAdd": 1965.0,
        "GrLivArea": 872.0,                  # Zillow: 872 sqft
        "FullBath": 2.0,                     # Zillow: 2 baths
        "HalfBath": 0.0,
        "BedroomAbvGr": 3.0,                 # Zillow: 3 beds
        "TotRmsAbvGrd": 6.0,
        "Fireplaces": 0.0,
        "GarageCars": 1.0,
        "GarageArea": 200.0,
        "NeighborhoodScore": 40.0,
        "CensusMedianValue": 68000.0,        # ZIP 48224 Detroit median
        "MedianIncomeK": 36.0,
        "OwnerOccupiedRate": 0.55,
        "SchoolDistrictRating": 3.5,
        "WalkScore": 62.0,
        "HOAFee": 0.0,                       # Zillow: $-- HOA
        "PricePerSqft": 60.8,               # 53000 / 872 ≈ $61/sqft (Zillow)
        "LandValue": 13250.0,               # 53000 × 0.25
        "PropertyType": "single_family",
        "City": "Detroit",
        "ZipCode": "48224",
        "State": "MI",
    },
    {
        "_address": "6601 Stagecoach Rd, Little Rock, AR 72204",
        "_label": "Luxury estate on 95 acres",
        "_zillow_price": 3311700,           # Zillow Zestimate
        "LotArea": 4138200.0,               # 95 acres × 43,560 sqft/acre
        "OverallQual": 9.0,
        "OverallCond": 7.0,
        "YearBuilt": 1928.0,               # Zillow: Built in 1928
        "YearRemodAdd": 2010.0,
        "GrLivArea": 9023.0,               # Zillow: 9,023 sqft
        "FullBath": 6.0,                   # Zillow: 6 baths
        "HalfBath": 0.0,
        "BedroomAbvGr": 6.0,              # Zillow: 6 beds
        "TotRmsAbvGrd": 16.0,
        "Fireplaces": 3.0,
        "GarageCars": 3.0,
        "GarageArea": 900.0,
        "NeighborhoodScore": 65.0,
        "CensusMedianValue": 134900.0,    # ZIP 72204 median from training CSV
        "MedianIncomeK": 52.0,
        "OwnerOccupiedRate": 0.62,
        "SchoolDistrictRating": 6.0,
        "WalkScore": 25.0,                # rural estate, very low walkability
        "HOAFee": 0.0,                    # Zillow: $-- HOA
        "PricePerSqft": 367.0,           # Zestimate / sqft (3311700 / 9023)
        "LandValue": 827925.0,           # Zestimate × 0.25
        "PropertyType": "luxury",
        "City": "Little Rock",
        "ZipCode": "72204",
        "State": "AR",
    },
    {
        "_address": "900 W 5th Ave, Anchorage, AK 99502",
        "_label": "Luxury estate",
        "_zillow_price": 595950,         # ZIP 99502 median from CSV
        "LotArea": 9200.0,
        "OverallQual": 9.0,
        "OverallCond": 8.0,
        "YearBuilt": 2015.0,
        "YearRemodAdd": 2022.0,
        "GrLivArea": 3800.0,
        "FullBath": 4.0,
        "HalfBath": 1.0,
        "BedroomAbvGr": 5.0,
        "TotRmsAbvGrd": 11.0,
        "Fireplaces": 2.0,
        "GarageCars": 3.0,
        "GarageArea": 720.0,
        "NeighborhoodScore": 90.0,
        "CensusMedianValue": 595950.0,   # ZIP 99502 median from CSV (was 480K — fixed)
        "MedianIncomeK": 115.0,
        "OwnerOccupiedRate": 0.71,
        "SchoolDistrictRating": 9.5,
        "WalkScore": 72.0,
        "HOAFee": 350.0,
        "PricePerSqft": 252.0,           # ZIP 99502 mean $/sqft from CSV (was 245 — fixed)
        "LandValue": 148988.0,           # 595950 * 0.25
        "PropertyType": "luxury",
        "City": "Anchorage",
        "ZipCode": "99502",
        "State": "AK",
    },
]

FEATURE_COLS = [k for k in properties[0] if not k.startswith("_")]

print()
print("=" * 72)
print("  MODEL PREDICTIONS — REAL ADDRESS TEST")
print("=" * 72)

for prop in properties:
    address = prop["_address"]
    label = prop["_label"]
    zillow = prop.get("_zillow_price")
    sqft = prop["GrLivArea"]
    beds = int(prop["BedroomAbvGr"])
    baths = int(prop["FullBath"])
    prop_type = prop["PropertyType"]

    df = pd.DataFrame([{k: prop[k] for k in FEATURE_COLS}])
    predicted = model.predict(df)[0]
    ppsf = predicted / sqft

    print(f"  Address : {address}")
    print(f"  Type    : {label} ({prop_type})  |  {int(sqft):,} sqft  |  {beds}bd/{baths}ba")
    print(f"  HOA/mo  : ${prop['HOAFee']:.0f}  |  Walk Score: {prop['WalkScore']:.0f}  |  School: {prop['SchoolDistrictRating']}/10")
    print(f"  --> Predicted Price : ${predicted:,.0f}  (${ppsf:.0f}/sqft)")
    if zillow:
        error = predicted - zillow
        pct = (error / zillow) * 100
        sign = "+" if error >= 0 else ""
        print(f"  --> Zillow Actual   : ${zillow:,}  |  Error: {sign}${error:,.0f}  ({sign}{pct:.1f}%)")
    print()

print("=" * 72)
