import requests

def osrm_distance_km(lat1, lon1, lat2, lon2):
    """
    Returns driving distance in km using OSRM public server.
    """
    p1 = f"{lon1},{lat1}"
    p2 = f"{lon2},{lat2}"

    url = f"http://router.project-osrm.org/route/v1/driving/{p1};{p2}"
    params = {
        "overview": "false",
        "alternatives": "false",
        "steps": "false"
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "routes" in data and len(data["routes"]) > 0:
        dist_m = data["routes"][0]["distance"]
        return dist_m / 1000.0  # km
    
    return None
