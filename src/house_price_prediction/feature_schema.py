from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


# Buyer-relevant features surfaced in every prediction and list response.
KEY_BUYER_FEATURES: tuple[str, ...] = (
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GrLivArea",
    "LotArea",
    "FullBath",
    "HalfBath",
    "YearBuilt",
    "GarageArea",
    "GarageCars",
    "OverallQual",
)

DEFAULT_PREDICTION_FEATURES: tuple[str, ...] = (
    # ── structural / physical ──────────────────────────────────────────
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "BasementSF",        # finished basement sqft (0 if no basement) — universal signal
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "Waterfront",        # 1 if waterfront property, 0 otherwise — 3x premium signal
    "ViewScore",         # scenic view quality 0–4 (0=none, 4=excellent) — up to 2.7x premium
    # ── property classification ────────────────────────────────────────
    "PropertyType",      # single_family | condo | townhouse | multifamily | luxury
    "HouseStyle",        # 1Story | 2Story | SFoyer | …
    "Neighborhood",      # postal-code neighborhood key used for location-aware routing
    # ── neighbourhood / market context (all sourced from US Census ACS at inference)
    # These features are populated at inference for ANY US address via the
    # live Census API, making the model generalise nationally.
    # "Neighborhood" (geographic string) was removed: it encoded KC-specific zip
    # codes as OHE columns during training, producing ~zero signal at inference
    # for any non-KC property and blocking national generalisation.
    "NeighborhoodScore",     # KNN score 0-100 using Census ACS median home value signal
    "CensusMedianValue",     # ACS B25077 tract median home value (USD)
    "MedianIncomeK",         # ACS B19013 tract median household income / 1000
    "OwnerOccupiedRate",     # fraction of owner-occupied units in census tract (0-1)
)


def align_feature_payload(
    expected_feature_names: Iterable[str],
    source_features: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_feature_names = list(expected_feature_names)
    if not ordered_feature_names:
        return dict(source_features)
    return {feature_name: source_features.get(feature_name) for feature_name in ordered_feature_names}