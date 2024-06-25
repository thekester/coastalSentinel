import cv2
import numpy as np
import json
from shapely.geometry import LineString, mapping
import geojson
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

# List of annotated image paths
image_paths = [
    "../newdata/annotated/dounambay2005_annotated.png",
    "../newdata/annotated/dounambayapril2013_annotated.png",
    "../newdata/annotated/dounambayapril2019_annotated.png",
    "../newdata/annotated/dounambayapril2024_annotated.png",
    "../newdata/annotated/dounambayjuly2011_annotated.png",
    "../newdata/annotated/dounambaymay2019_annotated.png",
    "../newdata/annotated/dounambaymay2021_annotated.png",
]

# Prepare to store GeoJSON features
features = []

# Process each image
for image_path in image_paths:
    # Extract the file name without the extension
    image_name = os.path.basename(image_path).replace('.png', '')
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Detect edges using Canny
    edges = cv2.Canny(binary, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black image
    coastline_image = np.zeros_like(gray)
    
    # Draw the contours in white
    cv2.drawContours(coastline_image, contours, -1, (255, 255, 255), 1)
    
    # Save the contour image
    coastline_image_path = f"../newdata/images/coastline/{image_name}_coastline.png"
    cv2.imwrite(coastline_image_path, coastline_image)
    
    # Convert the contours to GeoJSON
    for contour in contours:
        if len(contour) > 1:  # Ensure the contour has more than one point
            coordinates = contour[:, 0, :].tolist()
            linestring = LineString(coordinates)
            feature = {
                "type": "Feature",
                "geometry": mapping(linestring),
                "properties": {
                    "image": image_name  # File name without extension
                }
            }
            features.append(feature)

# Create a GeoJSON FeatureCollection
geojson_dict = {
    "type": "FeatureCollection",
    "features": features
}

# Save the GeoJSON data)
print(os.listdir())
geojson_file_path = "coastlines.json"
with open(geojson_file_path, 'w') as f:
    geojson.dump(geojson_dict, f)

# Display the paths to the saved images and the GeoJSON file
coastline_image_paths = [f"../newdata/images/coastline/{os.path.basename(image_path).replace('.png', '')}_coastline.png" for image_path in image_paths]
print(coastline_image_paths, geojson_file_path)
