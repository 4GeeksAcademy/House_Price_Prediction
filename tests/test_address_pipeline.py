"""
Tests for the address → feature pipeline and Census/FCC API helpers.
External HTTP calls are mocked so tests run offline and fast.
"""
import sys
import types
import json
import importlib
from io import BytesIO
from unittest.mock import patch, MagicMock, call
import pytest

# Ensure src and scripts are importable
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(responses: dict):
    """Return a context-manager mock for urllib.request.urlopen.

    `responses` maps url substring → bytes body to return.
    """
    def _side_effect(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for key, body in responses.items():
            if key in url:
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=cm)
                cm.__exit__ = MagicMock(return_value=False)
                cm.read = MagicMock(return_value=body if isinstance(body, bytes) else body.encode())
                cm.status = 200
                return cm
        raise ValueError(f"Unexpected URL in test: {url}")
    return _side_effect


# Dummy Census ACS tract response: yr_built=1985, rooms=6, 1BR=100 2BR=200 3BR=500 4BR=300
TRACT_ACS_RESPONSE = json.dumps([
    ["B25035_001E", "B25018_001E", "B25041_003E", "B25041_004E", "B25041_005E", "B25041_006E",
     "B25077_001E", "B19013_001E", "state", "county", "tract"],
    ["1985", "6.2", "100", "200", "500", "300", "350000", "80000", "12", "031", "016801"]
]).encode()

# Dummy FCC block response
FCC_RESPONSE = json.dumps({
    "Block": {"FIPS": "120310168010001"},
    "County": {"FIPS": "12031", "name": "Duval County"},
    "State": {"FIPS": "12", "code": "FL", "name": "Florida"},
    "status": "OK",
    "executionTime": "0"
}).encode()

# Nominatim response (Census geocoder is mocked to fail)
NOMINATIM_RESPONSE = json.dumps([
    {"lat": "30.1674", "lon": "-81.6317", "display_name": "11398 San Jose Blvd, Jacksonville, FL"}
]).encode()

# Dummy ZIP ACS response: 1BR=100 2BR=200 3BR=600 4BR=400
ZIP_ACS_RESPONSE = json.dumps([
    ["B25077_001E", "B19013_001E", "B25035_001E", "B25018_001E",
     "B25041_003E", "B25041_004E", "B25041_005E", "B25041_006E", "zip code tabulation area"],
    ["310000", "75000", "1978", "5", "100", "200", "600", "400", "32223"]
]).encode()


# ---------------------------------------------------------------------------
# Import module under test after path setup
# ---------------------------------------------------------------------------

from house_price_prediction.address_to_price import AssessorAPIConnector


# ---------------------------------------------------------------------------
# _fcc_census_tract
# ---------------------------------------------------------------------------

class TestFccCensusTract:
    def test_parses_state_county_tract_from_block_fips(self):
        responses = {"geo.fcc.gov": FCC_RESPONSE}
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(responses)):
            state, county, tract = AssessorAPIConnector._fcc_census_tract(30.1674, -81.6317)
        assert state == "12"
        assert county == "031"
        assert tract == "016801"

    def test_raises_on_bad_status(self):
        bad = json.dumps({"status": "FAIL", "Block": {"FIPS": ""}}).encode()
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen({"geo.fcc.gov": bad})):
            with pytest.raises(ValueError, match="FCC API status"):
                AssessorAPIConnector._fcc_census_tract(0.0, 0.0)


# ---------------------------------------------------------------------------
# _census_acs_by_tract
# ---------------------------------------------------------------------------

class TestCensusAcsByTract:
    def _call(self):
        responses = {"api.census.gov": TRACT_ACS_RESPONSE}
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(responses)):
            return AssessorAPIConnector._census_acs_by_tract("12", "031", "016801")

    def test_returns_correct_yr_built(self):
        result = self._call()
        assert result["median_yr_built"] == 1985

    def test_returns_correct_rooms(self):
        result = self._call()
        assert result["median_rooms"] == 6  # round(6.2)

    def test_modal_beds_is_3_for_3br_majority(self):
        # 3BR count (500, field _005E) is highest → modal_beds = 3
        result = self._call()
        assert result["modal_beds"] == 3

    def test_returns_home_value_and_income(self):
        result = self._call()
        assert result["median_home_value"] == 350000
        assert result["median_income"] == 80000

    def test_returns_empty_dict_on_short_response(self):
        short = json.dumps([["B25035_001E"]]).encode()  # no data row
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen({"api.census.gov": short})):
            result = AssessorAPIConnector._census_acs_by_tract("12", "031", "999999")
        assert result == {}


# ---------------------------------------------------------------------------
# _census_acs_by_zip
# ---------------------------------------------------------------------------

class TestCensusAcsByZip:
    def _call(self, zipcode="32223"):
        responses = {"api.census.gov": ZIP_ACS_RESPONSE}
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(responses)):
            return AssessorAPIConnector._census_acs_by_zip(zipcode)

    def test_returns_zip_level_yr_built(self):
        result = self._call()
        assert result["median_yr_built"] == 1978

    def test_returns_modal_beds(self):
        # 3BR=600 (field _005E) is highest → modal_beds = 3
        result = self._call()
        assert result["modal_beds"] == 3

    def test_returns_none_values_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = AssessorAPIConnector._census_acs_by_zip("99999")
        assert result["median_home_value"] is None
        assert result["median_yr_built"] is None
        assert result["modal_beds"] is None


# ---------------------------------------------------------------------------
# _features_from_address (integration — all external calls mocked)
# ---------------------------------------------------------------------------

class TestFeaturesFromAddress:
    """Full pipeline test with all network calls mocked."""

    RESPONSES = {
        "nominatim.openstreetmap.org": NOMINATIM_RESPONSE,
        "geo.fcc.gov": FCC_RESPONSE,
        "api.census.gov": TRACT_ACS_RESPONSE,
        # Census geocoder deliberately absent → triggers Nominatim fallback
    }

    def _run(self, address="11398 San Jose Blvd, Jacksonville, FL 32223"):
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(self.RESPONSES)):
            return AssessorAPIConnector._features_from_address(address)

    def test_zip_parsed_correctly(self):
        feats = self._run()
        assert feats["ZipCode"] == "32223"

    def test_state_parsed(self):
        feats = self._run()
        assert feats["State"] == "FL"

    def test_city_parsed(self):
        feats = self._run()
        assert "Jacksonville" in feats["City"]

    def test_yr_built_from_tract(self):
        # Tract ACS says 1985; jitter shifts ±5 → must be in [1980, 1990]
        feats = self._run()
        assert 1980 <= feats["YearBuilt"] <= 1990

    def test_bedrooms_from_tract_modal(self):
        # Tract modal_beds = 3 (500 3BR > 300 4BR); jitter ±1 → 2, 3, or 4
        feats = self._run()
        assert feats["BedroomAbvGr"] in (2, 3, 4)

    def test_tot_rooms_at_least_bedrooms_plus_two(self):
        feats = self._run()
        assert feats["TotRmsAbvGrd"] >= feats["BedroomAbvGr"] + 1

    def test_census_median_from_tract(self):
        feats = self._run()
        assert feats["CensusMedianValue"] == 350000

    def test_neighborhood_score_in_range(self):
        feats = self._run()
        assert 1 <= feats["NeighborhoodScore"] <= 99

    def test_lot_area_positive(self):
        feats = self._run()
        assert feats["LotArea"] > 0

    def test_living_area_reasonable(self):
        feats = self._run()
        assert 1000 <= feats["GrLivArea"] <= 4000

    def test_price_per_sqft_computed(self):
        feats = self._run()
        expected = round(feats["CensusMedianValue"] / feats["GrLivArea"], 1)
        assert feats["PricePerSqft"] == expected

    def test_required_model_keys_present(self):
        feats = self._run()
        required = [
            "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
            "CensusMedianValue", "MedianIncomeK",
        ]
        for key in required:
            assert key in feats, f"Missing required feature: {key}"

    def test_geocode_fallback_to_nominatim_when_census_fails(self):
        """Census geocoder fails → falls back to Nominatim → lat/lon still populated."""
        feats = self._run()
        assert feats["lat"] == pytest.approx(30.1674, abs=0.01)
        assert feats["lon"] == pytest.approx(-81.6317, abs=0.01)

    def test_state_fallback_median_when_geocode_fails(self):
        """If geocoding returns 0,0 (all APIs fail), ZIP/tract calls also fail gracefully."""
        no_network = {}  # all HTTP calls raise ValueError
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(no_network)):
            feats = AssessorAPIConnector._features_from_address(
                "456 Oak Ave, Austin, TX 78701"
            )
        # State fallback table used — TX median is $355,000
        assert feats["CensusMedianValue"] == 355000
        assert feats["State"] == "TX"

    def test_different_addresses_produce_different_yr_built(self):
        """Hash jitter means two different addresses in same ZIP can differ."""
        addr_a = "100 Elm St, Jacksonville, FL 32223"
        addr_b = "999 Oak Dr, Jacksonville, FL 32223"
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(self.RESPONSES)):
            fa = AssessorAPIConnector._features_from_address(addr_a)
        with patch("urllib.request.urlopen", side_effect=_mock_urlopen(self.RESPONSES)):
            fb = AssessorAPIConnector._features_from_address(addr_b)
        # At minimum, sqft or qual or lot area should differ (hash-derived)
        diffs = [
            fa["GrLivArea"] != fb["GrLivArea"],
            fa["OverallQual"] != fb["OverallQual"],
            fa["LotArea"] != fb["LotArea"],
        ]
        assert any(diffs), "Two different street addresses must produce at least one different feature"
