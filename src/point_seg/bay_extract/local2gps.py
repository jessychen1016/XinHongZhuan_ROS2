import math
from pyproj import Proj, Transformer


# the x the the forward of the ship while the y is the right of the ship
def ship_local_to_gps(x0, y0, long0, lat0, heading_deg):
    # Set up a UTM projection for the reference longitude/latitude
    utm_proj = Proj(proj='utm', zone=int((long0 + 180) / 6) + 1, ellps='WGS84', preserve_units=False)
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    transformer_to_utm = Transformer.from_proj(wgs84_proj, utm_proj, always_xy=True)
    transformer_from_utm = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)

    # Convert ship center (long0, lat0) to UTM
    x_center, y_center = transformer_to_utm.transform(long0, lat0)

    # Heading in radians (convert from degrees, clockwise from North)
    heading_rad = math.radians(heading_deg)

    # Ship axes: x is right, y is forward
    # True north is along +Y
    # To rotate local (x0, y0) to north-east axes:
    #   x_east = x0 * cos(a) - y0 * sin(a)
    #   y_north = x0 * sin(a) + y0 * cos(a)
    dx = x0 * math.sin(heading_rad) + y0 * math.cos(heading_rad)
    dy = x0 * math.cos(heading_rad) - y0 * math.sin(heading_rad)

    # Get UTM coordinates of the point
    x_point = x_center + dx
    y_point = y_center + dy

    # Convert back to lon/lat
    lon, lat = transformer_from_utm.transform(x_point, y_point)
    return lon, lat

# Example usage:
if __name__ == "__main__":
    # Example: point (2, 10) meters in ship frame, ship at (120.2, 35.5), heading 30 deg clockwise from North
    x0, y0 = 2, 10
    long0, lat0 = 120.255643, 35.55567
    heading_deg = 45
    lon, lat = ship_local_to_gps(x0, y0, long0, lat0, heading_deg)
    print("GPS coordinate:", lon, lat)