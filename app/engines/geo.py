from math import radians, sin, cos, sqrt, atan2

class GeoEngine:
    def distance_km(self, a, b) -> float:
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [a.latitude, a.longitude, b.latitude, b.longitude])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        x = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        return 2 * R * atan2(sqrt(x), sqrt(1 - x))

    def score(self, user_location, product_location) -> float:
        if not user_location or not product_location:
            return 0.5
        d = self.distance_km(user_location, product_location)
        return 1.0 / (1.0 + d / 10.0)
