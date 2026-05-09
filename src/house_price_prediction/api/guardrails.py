from __future__ import annotations

import re

from house_price_prediction.config import Settings
from house_price_prediction.domain.contracts.prediction_contracts import AddressPayload

_PO_BOX_PATTERN = re.compile(r"\bP\s*\.?\s*O\s*\.?\s*BOX\b", re.IGNORECASE)
_US_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")


class RequestGuardrailError(ValueError):
    pass


def validate_address_payload(payload: AddressPayload, settings: Settings) -> None:
    if _PO_BOX_PATTERN.search(payload.address_line_1):
        raise RequestGuardrailError("Street address is required; PO boxes are not supported.")

    if _uses_us_only_providers(settings) and payload.country.strip().upper() != "US":
        raise RequestGuardrailError(
            "Configured public providers currently support only US addresses."
        )

    if payload.country.strip().upper() == "US" and not _US_ZIP_PATTERN.match(payload.postal_code.strip()):
        raise RequestGuardrailError(
            "postal_code must be a valid US ZIP code (5 digits, or 5+4 with hyphen). "
            f"Got: {payload.postal_code!r}"
        )


def _uses_us_only_providers(settings: Settings) -> bool:
    # The "free" and "free-fallback" provider tiers both use the Census ACS API
    # which only covers US addresses.  Nominatim (used internally in these tiers)
    # can geocode international addresses but Census property enrichment will fail
    # for non-US coordinates, so we gate early at the request level.
    # The "fake" provider has no geographic restriction and is excluded deliberately.
    geo = settings.geocoding_provider.strip().lower()
    prop = settings.property_data_provider.strip().lower()
    return geo.startswith("free") or prop.startswith("free")