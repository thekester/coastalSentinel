import json
import numpy as np
from shapely.geometry import LineString
import os
print("os")
# Lists all items in the current directory
directories = os.listdir()

# Filter to find a specific folder
target_directory = next((dir for dir in directories if "coastalSentinel" in dir), None)

if target_directory:
    # Change working directory
    os.chdir(target_directory)
    print(f"Path changed to: {os.getcwd()}")
    # If you then want to go to the 'src' subfolder
    if 'src' in os.listdir():
        os.chdir('src')
        print(f"Path changed to subfolder 'src': {os.getcwd()}")
    else:
        print("'src' subfolder does not exist.")
else:
    print("Directory 'coastalSentinel' not find.")

def calculate_distances(coords1, coords2):
    distances = []
    for c1, c2 in zip(coords1, coords2):
        distance = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        distances.append(distance)
    return distances

# Load GeoJSON data
with open('coastlines.json', 'r') as f:
    data = json.load(f)

# Extract coordinates for different years
coastlines = {}
for feature in data['features']:
    year = feature['properties']['image']  # Assuming 'image' contains year info
    coords = feature['geometry']['coordinates']
    coastlines[year] = coords

# Calculate distances between consecutive years
years = sorted(coastlines.keys())
distance_results = {}

for i in range(len(years) - 1):
    year1, year2 = years[i], years[i + 1]
    coords1, coords2 = coastlines[year1], coastlines[year2]
    
    # Ensure both lists have the same length for comparison
    min_length = min(len(coords1), len(coords2))
    coords1, coords2 = coords1[:min_length], coords2[:min_length]
    
    distances = calculate_distances(coords1, coords2)
    distance_results[f"{year1}-{year2}"] = distances

# Save the results to a JSON file
with open('distance_results.json', 'w') as f:
    json.dump(distance_results, f)
