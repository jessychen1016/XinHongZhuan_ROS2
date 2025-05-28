import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj, Transformer

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

def plot_ship_points():
    # Ship parameters
    ship_long, ship_lat = 120.255643, 35.55567
    heading_deg = 75
    
    # Local points in ship coordinates (x,y)
    points = [(2,2), (2,6), (10,2), (10,6)]
    
    # Calculate GPS coordinates for each point
    gps_points = []
    for x, y in points:
        lon, lat = ship_local_to_gps(x, y, ship_long, ship_lat, heading_deg)
        gps_points.append((lon, lat))
    
    # Create plot
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    
    # Plot ship center
    ax.plot(ship_long, ship_lat, 'ro', markersize=10, label='Ship Center')
    
    # Plot ship axes (x and y)
    axis_length = 1  # degrees 
    # X-axis 
    x_end_lon, x_end_lat = ship_local_to_gps(axis_length, 0, ship_long, ship_lat, heading_deg)
    ax.plot([ship_long, x_end_lon], [ship_lat, x_end_lat], 'r-', linewidth=2, label='X-axis')
    # Y-axis
    y_end_lon, y_end_lat = ship_local_to_gps(0, axis_length, ship_long, ship_lat, heading_deg)
    ax.plot([ship_long, y_end_lon], [ship_lat, y_end_lat], 'g-', linewidth=2, label='Y-axis')

    # Plot converted points and calculate distances
    for i, (lon1, lat1) in enumerate(gps_points):
        ax.plot(lon1, lat1, 'bo', markersize=8)
        ax.text(lon1, lat1, f'Point {i+1}', fontsize=12, 
                transform=ccrs.PlateCarree())
        
        # Calculate distances to other points
        for j, (lon2, lat2) in enumerate(gps_points[i+1:], start=i+1):
            # Haversine distance calculation
            dlon = math.radians(lon2 - lon1)
            dlat = math.radians(lat2 - lat1)
            a = (math.sin(dlat/2)**2 + 
                math.cos(math.radians(lat1)) * 
                math.cos(math.radians(lat2)) * 
                math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = 6371000 * c  # Earth radius in meters
            
            # Draw distance line and label
            ax.plot([lon1, lon2], [lat1, lat2], 'k:', alpha=0.5)
            mid_lon = (lon1 + lon2)/2
            mid_lat = (lat1 + lat2)/2
            ax.text(mid_lon, mid_lat, f'{distance:.1f}m', 
                   fontsize=10, ha='center', va='bottom',
                   transform=ccrs.PlateCarree())
    
    # Set extent around points
    min_lon = min(lon for lon, lat in gps_points) - 0.01
    max_lon = max(lon for lon, lat in gps_points) + 0.01
    min_lat = min(lat for lon, lat in gps_points) - 0.01
    max_lat = max(lat for lon, lat in gps_points) + 0.01
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    
    # Add gridlines and labels
    ax.gridlines(draw_labels=True)
    ax.legend()
    plt.title('Ship Points in GPS Coordinates')
    
    # Save and show plot
    plt.savefig('ship_points_plot.png')
    plt.show()
    
    return gps_points

if __name__ == "__main__":
    gps_coords = plot_ship_points()
    print("Calculated GPS coordinates:")
    for i, (lon, lat) in enumerate(gps_coords):
        print(f"Point {i+1}: ({lon:.6f}, {lat:.6f})")
