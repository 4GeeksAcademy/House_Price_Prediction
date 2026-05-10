from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from threading import Lock

from house_price_prediction.domain.contracts.prediction_contracts import (
    NormalizedAddress,
    ProviderResponseContract,
)
from house_price_prediction.infrastructure.providers.property_type_classifier import (
    classify_property_type,
)


class HeuristicPropertyDataClient:
    _LOCAL_PROFILE_LOCK = Lock()
    _LOCAL_PROFILE_BY_ZIP: dict[str, dict[str, float]] | None = None
    _LOCAL_PROFILE_SOURCE: str | None = None
    _LOCAL_PROFILE_MIN_SAMPLES = 5

    _STATE_NAME_TO_ABBR: dict[str, str] = {
        "ALABAMA": "AL",
        "ALASKA": "AK",
        "ARIZONA": "AZ",
        "ARKANSAS": "AR",
        "CALIFORNIA": "CA",
        "COLORADO": "CO",
        "CONNECTICUT": "CT",
        "DELAWARE": "DE",
        "DISTRICT OF COLUMBIA": "DC",
        "FLORIDA": "FL",
        "GEORGIA": "GA",
        "HAWAII": "HI",
        "IDAHO": "ID",
        "ILLINOIS": "IL",
        "INDIANA": "IN",
        "IOWA": "IA",
        "KANSAS": "KS",
        "KENTUCKY": "KY",
        "LOUISIANA": "LA",
        "MAINE": "ME",
        "MARYLAND": "MD",
        "MASSACHUSETTS": "MA",
        "MICHIGAN": "MI",
        "MINNESOTA": "MN",
        "MISSISSIPPI": "MS",
        "MISSOURI": "MO",
        "MONTANA": "MT",
        "NEBRASKA": "NE",
        "NEVADA": "NV",
        "NEW HAMPSHIRE": "NH",
        "NEW JERSEY": "NJ",
        "NEW MEXICO": "NM",
        "NEW YORK": "NY",
        "NORTH CAROLINA": "NC",
        "NORTH DAKOTA": "ND",
        "OHIO": "OH",
        "OKLAHOMA": "OK",
        "OREGON": "OR",
        "PENNSYLVANIA": "PA",
        "RHODE ISLAND": "RI",
        "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD",
        "TENNESSEE": "TN",
        "TEXAS": "TX",
        "UTAH": "UT",
        "VERMONT": "VT",
        "VIRGINIA": "VA",
        "WASHINGTON": "WA",
        "WEST VIRGINIA": "WV",
        "WISCONSIN": "WI",
        "WYOMING": "WY",
    }
    _STATE_MEDIAN_HOME_VALUE: dict[str, int] = {
        "WA": 575000, "CA": 790000, "HI": 835000, "MA": 620000, "CO": 580000,
        "OR": 480000, "NJ": 520000, "NY": 550000, "UT": 520000, "AZ": 430000,
        "FL": 415000, "GA": 330000, "TX": 355000, "NC": 330000, "IL": 315000,
        "OH": 265000, "PA": 295000, "MI": 265000, "TN": 335000, "VA": 450000,
        "MD": 435000, "NV": 430000, "ID": 430000, "MN": 335000, "WI": 290000,
        "IN": 250000, "MO": 270000, "SC": 320000, "AL": 255000, "KY": 255000,
        "OK": 235000, "AR": 235000, "MS": 215000, "IA": 245000, "KS": 255000,
        "NE": 280000, "SD": 260000, "ND": 265000, "NM": 320000, "LA": 260000,
        "MT": 415000, "WV": 200000, "AK": 410000, "DC": 720000, "CT": 435000,
        "RI": 470000, "NH": 455000, "VT": 420000, "ME": 400000, "DE": 395000,
        "WY": 355000,
    }
    _STATE_MEDIAN_INCOME: dict[str, int] = {
        "WA": 95000, "CA": 91000, "HI": 88000, "MA": 96000, "CO": 89000,
        "OR": 78000, "NJ": 97000, "NY": 82000, "UT": 85000, "AZ": 72000,
        "FL": 67000, "GA": 71000, "TX": 73000, "NC": 68000, "IL": 75000,
        "OH": 66000, "PA": 72000, "MI": 67000, "TN": 66000, "VA": 90000,
        "MD": 98000, "NV": 70000, "ID": 70000, "MN": 84000, "WI": 72000,
        "IN": 65000, "MO": 66000, "SC": 64000, "AL": 59000, "KY": 60000,
        "OK": 60000, "AR": 57000, "MS": 52000, "IA": 68000, "KS": 67000,
        "NE": 70000, "SD": 65000, "ND": 71000, "NM": 58000, "LA": 57000,
        "MT": 65000, "WV": 51000, "AK": 82000, "DC": 101000, "CT": 90000,
        "RI": 77000, "NH": 92000, "VT": 72000, "ME": 68000, "DE": 76000,
        "WY": 68000,
    }

    def fetch_property_features(
        self,
        normalized_address: NormalizedAddress,
    ) -> ProviderResponseContract:
        state = self._normalize_state(normalized_address.state)
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

        market_anchor = self._state_market_anchor(state, latitude, longitude, quality_score)
        income_anchor = self._STATE_MEDIAN_INCOME.get(state, 70000)

        # Keep the fallback profile conservative for a typical single-family home.
        overall_qual = self._clamp(
            int(round(6 + (quality_score - 0.5) * 1.0 + (1.0 if market_anchor >= 700000 else 0.0))),
            5,
            8,
        )
        year_built = self._clamp(int(round(1980 + build_epoch * 30)), 1950, 2022)
        gr_liv_area = self._clamp(
            int(round(1700 + (quality_score - 0.5) * 250.0 + (1.0 - urban_score) * 100.0)),
            1400,
            2300,
        )
        lot_area = self._clamp(
            int(round(6800 + (1.0 - urban_score) * 1600.0 + max(0.0, market_anchor - 500000.0) / 500.0)),
            3500,
            13000,
        )

        total_rooms = self._clamp(int(round(gr_liv_area / 300.0)), 5, 10)
        bedrooms = self._clamp(3 if gr_liv_area >= 1600 else 2, 2, 4)
        full_bath = self._clamp(2 if gr_liv_area >= 1500 else 1, 1, 3)
        half_bath = 1 if quality_score >= 0.7 and gr_liv_area >= 2000 else 0
        garage_cars = self._clamp(
            1 + int(round((1.0 - urban_score) * 0.3 + max(0.0, market_anchor - 750000.0) / 500000.0)),
            1,
            3,
        )
        garage_area = garage_cars * 240
        overall_cond = self._clamp(5 + int(round((1.0 - quality_score) * 1.5)), 4, 8)

        # Heuristic-derived neighborhood economic estimates anchored to the state market.
        est_median_value = int(round(market_anchor * (0.99 + (quality_score - 0.5) * 0.04)))
        est_median_income_k = round((income_anchor * (0.98 + (quality_score - 0.5) * 0.05)) / 1000.0, 1)
        est_owner_rate = round(
            max(0.35, min(0.82, 0.50 + (quality_score - 0.5) * 0.12 + (1.0 - urban_score) * 0.06)),
            3,
        )

        payload = {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": min(2024, year_built + 8 + round(self._fraction(seed, "remodel") * 12)),
            "GrLivArea": gr_liv_area,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "Fireplaces": 1 if quality_score > 0.55 else 0,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "BasementSF": 0,
            "Waterfront": 0,
            "ViewScore": 0,
            "Neighborhood": normalized_address.postal_code or postal_code,
            "HouseStyle": "2Story" if gr_liv_area >= 1800 and urban_score >= 0.35 else "1Story",
            # ── market context features ─────────────────────────────
            "CensusMedianValue": est_median_value,
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

        local_profile = self._lookup_local_profile(postal_code)
        if local_profile is not None:
            payload.update(
                {
                    "LotArea": self._clamp(int(round(local_profile.get("LotArea", payload["LotArea"]))), 3000, 30000),
                    "OverallQual": self._clamp(int(round(local_profile.get("OverallQual", payload["OverallQual"]))), 3, 10),
                    "OverallCond": self._clamp(int(round(local_profile.get("OverallCond", payload["OverallCond"]))), 3, 9),
                    "YearBuilt": self._clamp(int(round(local_profile.get("YearBuilt", payload["YearBuilt"]))), 1900, 2024),
                    "GrLivArea": self._clamp(int(round(local_profile.get("GrLivArea", payload["GrLivArea"]))), 800, 6000),
                    "FullBath": self._clamp(int(round(local_profile.get("FullBath", payload["FullBath"]))), 1, 5),
                    "HalfBath": self._clamp(int(round(local_profile.get("HalfBath", payload["HalfBath"]))), 0, 3),
                    "BedroomAbvGr": self._clamp(int(round(local_profile.get("BedroomAbvGr", payload["BedroomAbvGr"]))), 1, 7),
                    "TotRmsAbvGrd": self._clamp(int(round(local_profile.get("TotRmsAbvGrd", payload["TotRmsAbvGrd"]))), 3, 15),
                    "GarageCars": self._clamp(int(round(local_profile.get("GarageCars", payload["GarageCars"]))), 0, 5),
                    "GarageArea": self._clamp(int(round(local_profile.get("GarageArea", payload["GarageArea"]))), 0, 1400),
                }
            )

            payload["HouseStyle"] = "2Story" if payload["GrLivArea"] >= 1800 and urban_score >= 0.35 else "1Story"
            payload["feature_source"] = "local_data_prior_with_heuristic"
            payload["feature_provenance"] = {
                "strategy": "local_data_prior_with_heuristic",
                "providers": ["heuristic_property_data"],
                "derived_from": ["postal_code", "csv_training_data"],
                "postal_code": postal_code,
                "sample_count": int(local_profile.get("sample_count", 0.0)),
                "source_file": self._LOCAL_PROFILE_SOURCE,
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

    @classmethod
    def _normalize_state(cls, state: str | None) -> str:
        raw_state = (state or "").strip().upper()
        if len(raw_state) == 2:
            return raw_state
        return cls._STATE_NAME_TO_ABBR.get(raw_state, raw_state)

    @classmethod
    def _state_market_anchor(cls, state: str, latitude: float, longitude: float, quality_score: float) -> float:
        """Return a state-level market anchor, or a geographic fallback when state is unknown."""
        state_value = cls._STATE_MEDIAN_HOME_VALUE.get(state)
        if state_value is not None:
            return float(state_value)
        return float(cls._estimate_median_value(latitude, longitude, quality_score))

    @staticmethod
    def _estimate_median_value(
        latitude: float,
        longitude: float,
        quality_score: float,
    ) -> int:
        """Fallback geographic median when a state code is unavailable."""
        lat, lon = latitude, longitude

        if 18 <= lat <= 23 and -163 <= lon <= -154:
            lo, hi = 600_000, 1_200_000
        elif lat > 54:
            lo, hi = 280_000, 600_000
        elif 32 <= lat <= 49 and lon < -114:
            lo, hi = 450_000, 1_100_000
        elif lat >= 37 and lon >= -80:
            lo, hi = 350_000, 900_000
        elif 31 <= lat <= 49 and -115 <= lon <= -101:
            lo, hi = 280_000, 700_000
        elif 24 <= lat <= 37 and -85 <= lon <= -75:
            lo, hi = 250_000, 600_000
        elif 25 <= lat <= 36 and -107 <= lon <= -85:
            lo, hi = 150_000, 400_000
        elif 37 <= lat <= 49 and -97 <= lon <= -80:
            lo, hi = 150_000, 380_000
        elif 37 <= lat <= 49 and -105 <= lon <= -97:
            lo, hi = 130_000, 300_000
        else:
            lo, hi = 150_000, 500_000

        return int(lo + quality_score * (hi - lo))

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    @classmethod
    def _lookup_local_profile(cls, postal_code: str) -> dict[str, float] | None:
        profile_by_zip = cls._load_local_profiles()
        if not profile_by_zip:
            return None
        zip5 = (postal_code or "")[:5]
        if len(zip5) != 5 or not zip5.isdigit():
            return None
        return profile_by_zip.get(zip5)

    @classmethod
    def _load_local_profiles(cls) -> dict[str, dict[str, float]]:
        if cls._LOCAL_PROFILE_BY_ZIP is not None:
            return cls._LOCAL_PROFILE_BY_ZIP

        with cls._LOCAL_PROFILE_LOCK:
            if cls._LOCAL_PROFILE_BY_ZIP is not None:
                return cls._LOCAL_PROFILE_BY_ZIP

            data_file = cls._resolve_local_training_data_file()
            if data_file is None:
                cls._LOCAL_PROFILE_BY_ZIP = {}
                cls._LOCAL_PROFILE_SOURCE = None
                return cls._LOCAL_PROFILE_BY_ZIP

            field_names = (
                "LotArea",
                "OverallQual",
                "OverallCond",
                "YearBuilt",
                "GrLivArea",
                "FullBath",
                "HalfBath",
                "BedroomAbvGr",
                "TotRmsAbvGrd",
                "GarageCars",
                "GarageArea",
            )
            sums: dict[str, dict[str, float]] = {}
            counts: dict[str, int] = {}

            try:
                with data_file.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        zip_code = str(record.get("Neighborhood") or "").strip()[:5]
                        if len(zip_code) != 5 or not zip_code.isdigit():
                            continue

                        bucket = sums.setdefault(zip_code, {})
                        row_has_numeric = False
                        for field_name in field_names:
                            value = record.get(field_name)
                            if isinstance(value, int | float):
                                bucket[field_name] = bucket.get(field_name, 0.0) + float(value)
                                row_has_numeric = True

                        if row_has_numeric:
                            counts[zip_code] = counts.get(zip_code, 0) + 1
            except OSError:
                cls._LOCAL_PROFILE_BY_ZIP = {}
                cls._LOCAL_PROFILE_SOURCE = None
                return cls._LOCAL_PROFILE_BY_ZIP

            profiles: dict[str, dict[str, float]] = {}
            for zip_code, count in counts.items():
                if count < cls._LOCAL_PROFILE_MIN_SAMPLES:
                    continue
                source = sums.get(zip_code, {})
                if not source:
                    continue
                profile = {field_name: source.get(field_name, 0.0) / float(count) for field_name in field_names}
                profile["sample_count"] = float(count)
                profiles[zip_code] = profile

            cls._LOCAL_PROFILE_BY_ZIP = profiles
            cls._LOCAL_PROFILE_SOURCE = str(data_file)
            return cls._LOCAL_PROFILE_BY_ZIP

    @staticmethod
    def _resolve_local_training_data_file() -> Path | None:
        current = Path(__file__).resolve()
        repo_root = current.parents[4]
        candidate = repo_root / "data" / "processed" / "csv_training_data.jsonl"
        if candidate.exists():
            return candidate
        return None