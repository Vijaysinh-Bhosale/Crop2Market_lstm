import time
import requests
from geopy.geocoders import Nominatim

# ---------- Geocoder setup (Nominatim) ----------
_geolocator = Nominatim(user_agent="crop2market_pincode_router")

def geocode_pincode(pincode: str, country: str = "India"):
    """
    Convert a pincode to (lat, lon) using Nominatim (OpenStreetMap) API.
    This is an ONLINE API call. Do not spam it in tight loops.
    """
    query = f"{pincode}, {country}"
    loc = _geolocator.geocode(query, exactly_one=True, timeout=10)

    # Be polite to the free Nominatim service: small delay between calls
    time.sleep(1.0)

    if loc is None:
        raise ValueError(f"Could not geocode pincode {pincode}")
    return float(loc.latitude), float(loc.longitude)


# ---------- OSRM routing ----------
OSRM_BASE_URL = "http://router.project-osrm.org"

def osrm_driving_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Call OSRM public API to get DRIVING distance in km between two coordinates.
    Raises an error if something goes wrong so you notice.
    """
    # OSRM expects lon,lat
    p1 = f"{lon1},{lat1}"
    p2 = f"{lon2},{lat2}"

    url = f"{OSRM_BASE_URL}/route/v1/driving/{p1};{p2}"
    params = {
        "overview": "false",
        "alternatives": "false",
        "steps": "false"
    }

    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    if resp.status_code != 200:
        raise RuntimeError(f"OSRM HTTP {resp.status_code}: {data}")

    if "routes" not in data or not data["routes"]:
        raise RuntimeError(f"OSRM no routes found: {data}")

    dist_m = data["routes"][0]["distance"]  # meters
    return dist_m / 1000.0                  # km


# ---------- High-level helper ----------
def driving_distance_km_between_pincodes(src_pincode: str, dst_pincode: str, country: str = "India") -> float:
    """
    1) Geocode both pincodes via Nominatim API
    2) Call OSRM API to get driving distance
    Returns distance in km.
    """
    lat1, lon1 = geocode_pincode(src_pincode, country)
    lat2, lon2 = geocode_pincode(dst_pincode, country)

    return osrm_driving_distance_km(lat1, lon1, lat2, lon2)
