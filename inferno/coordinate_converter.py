import numpy as np

# -122.49837646484295,40.73969482968265
lat = 40.73969482968265
lon = -122.49837646484295

width_km = 36.576
height_km = 24.384

lat_deg_km = 111.0
lon_deg_km = 111.0 * np.cos(np.radians(lat))
w_deg = width_km / lon_deg_km
h_deg = height_km / lat_deg_km

f = lon - w_deg / 2
s = lat - h_deg / 2
t = lon + w_deg / 2
fo = lat + h_deg / 2
# region = [
#     [lon - w_deg / 2, lat - h_deg / 2],
#     [lon + w_deg / 2, lat - h_deg / 2],
#     [lon + w_deg / 2, lat + h_deg / 2],
#     [lon - w_deg / 2, lat + h_deg / 2]
# ]

print([f,s,t,fo])