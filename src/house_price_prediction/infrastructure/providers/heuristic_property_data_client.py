from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)


class HeuristicPropertyDataClient:
    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        postal_code = normalized_address.postal_code or "00000"
        latitude = normalized_address.latitude or 39.5
        longitude = normalized_address.longitude or -98.35
        seed = (
            f"{normalized_address.formatted_address}|{postal_code}|"
            f"{latitude:.4f}|{longitude:.4f}"
        )

        urban_score = self._fraction(seed, "urban_score")
        quality_score = self._fraction(seed, "quality_score")
        build_epoch = self._fraction(seed, "build_epoch")

        overall_qual = 5 + round(quality_score * 4)
        total_rooms = 5 + round(urban_score * 4)
        garage_cars = 1 + round((1 - urban_score) * 2)
        year_built = 1965 + round(build_epoch * 57)
        lot_area = 5000 + round((1 - urban_score) * 9000)

        # Heuristic-derived neighbourhood economic estimates
        # Higher quality_score → higher estimated market value
        est_median_value = int(80_000 + quality_score * 500_000)   # $80k–$580k range
        est_median_income_k = round(30.0 + quality_score * 70.0, 1)  # $30k–$100k / 1000
        # Urban areas trend toward renting; suburban toward owning
        est_owner_rate = round(max(0.2, 0.9 - urban_score * 0.55), 3)

        payload = {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": 5 + round(self._fraction(seed, "condition") * 3),
            "YearBuilt": year_built,
            "YearRemodAdd": min(2024, year_built + 8 + round(self._fraction(seed, "remodel") * 12)),
            "GrLivArea": 1100 + (total_rooms * 220),
            "FullBath": 1 + round(self._fraction(seed, "full_bath") * 2),
            "HalfBath": round(self._fraction(seed, "half_bath")),
            "BedroomAbvGr": max(2, total_rooms // 2),
            "TotRmsAbvGrd": total_rooms,
            "Fireplaces": round(self._fraction(seed, "fireplaces") * 2),
            "GarageCars": garage_cars,
            "GarageArea": garage_cars * 240,
            "HouseStyle": "2Story" if urban_score >= 0.55 else "1Story",
            # ── market context features ─────────────────────────────
            "CensusMedianValue": self._estimate_median_value(
                latitude, longitude, quality_score
            ),
            "MedianIncomeK": est_median_income_k,
            "OwnerOccupiedRate": est_owner_rate,
            # NeighborhoodScore: continuous 0-100 scale (not 0-10)
            "NeighborhoodScore": round(quality_score * 100.0, 2),
            "feature_source": "heuristic",
            "feature_provenance": {
                "strategy": "heuristic",
                "providers": ["heuristic_property_data"],
                "derived_from": ["formatted_address", "postal_code", "coordinates"],
            },
        }
        payload["PropertyType"] = classify_property_type(payload)

        return ProviderResponseContract(
            provider_name="heuristic_property_data",
            status="success",
            payload=payload,
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _fraction(seed: str, name: str) -> float:
        digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    @staticmethod
    def _estimate_median_value(
        latitude: float,
        longitude: float,
        quality_score: float,
    ) -> int:
        """Estimate census-tract median home value from lat/lon + quality.

        Uses US geographic market tiers (derived from Census ACS5 regional
        medians) to anchor the range, then adjusts within the tier by
        ``quality_score`` (0–1).  This is substantially more accurate than
        the old purely hash-based $80k–$580k range.

        Tier anchors (approximate ACS5 2022 US regional medians):
          Pacific West  (CA/OR/WA):                $450k – $1 100k
          Northeast     (NY/NJ/CT/MA/DC area):     $350k –  $900k
          Mountain West (CO/UT/NV/AZ/ID/MT/WY):   $280k –  $700k
          Hawaii:                                  $600k –  $1 200k
          Alaska:                                  $280k –  $600k
          South/South-Atlantic (FL/VA/MD/TX):      $250k –  $600k
          Midwest South (OK/AR/LA/MS/AL):          $120k –  $300k
          Midwest North (MN/WI/IL/MI/OH/IN/MO):   $150k –  $380k
          Plains (ND/SD/NE/KS):                    $130k –  $300k
          Default (generic US):                    $150k –  $500k
        """
        lat, lon = latitude, longitude

        # Hawaii (major islands: lat 18–23, lon –162 to –154)
        if 18 <= lat <= 23 and -163 <= lon <= -154:
            lo, hi = 600_000, 1_200_000
        # Alaska
        elif lat > 54:
            lo, hi = 280_000, 600_000
        # Pacific Coast: CA, OR, WA (lon west of -114, lat 32–49)
        elif 32 <= lat <= 49 and lon < -114:
            lo, hi = 450_000, 1_100_000
        # Northeast corridor: roughly ME/NH/VT/MA/RI/CT/NY/NJ/PA/DE/MD/DC
        elif lat >= 37 and lon >= -80:
            lo, hi = 350_000, 900_000
        # Mountain West: AZ/NM/CO/UT/NV/ID/MT/WY
        elif 31 <= lat <= 49 and -115 <= lon <= -101:
            lo, hi = 280_000, 700_000
        # South Atlantic / Mid-South: FL, GA, SC, NC, VA, WV, TN, KY
        elif 24 <= lat <= 37 and -85 <= lon <= -75:
            lo, hi = 250_000, 600_000
        # Texas + southern plains: TX, OK, LA, AR, MS, AL
        elif 25 <= lat <= 36 and -107 <= lon <= -85:
            lo, hi = 150_000, 400_000
        # Midwest: MN, WI, IL, MI, OH, IN, IA, MO
        elif 37 <= lat <= 49 and -97 <= lon <= -80:
            lo, hi = 150_000, 380_000
        # Northern plains: ND, SD, NE, KS, MN western
        elif 37 <= lat <= 49 and -105 <= lon <= -97:
            lo, hi = 130_000, 300_000
        else:
            lo, hi = 150_000, 500_000

        return int(lo + quality_score * (hi - lo))