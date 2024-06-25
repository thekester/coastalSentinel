# Coastal Sentinel 🌊🛰️

## Overview
This repository contains various scripts experimenting with machine learning algorithms to detect coastlines in images. The goal is to identify the most effective algorithms for coastal analysis.

## License 
License details will be updated soon.

## Code Structure
Explore the `src` folder for the following scripts:

- `color_segmentation_image_generate.ipynb`: Script for generating images using color segmentation techniques.
- `combined_color_segmentation_edge_detection_generate_label.ipynb`: This script attempts to combine color segmentation with edge detection to automatically create labeled images.

### Scripts Based on Conor O'Sullivan's 2022 Works
The following notebooks are inspired by the work of Conor O'Sullivan and are located in the `src` folder:
- `SWED_exploration.ipynb`: Initial analysis of the SWED dataset focusing on data quality.
- `edge_detection.ipynb`: Implementation of various edge detection algorithms (Canny, Sobel, Scharr, Prewitt) on the SWED dataset. This notebook includes all code for figures presented in the related conference paper.

## Annotations
Annotations are performed with GIMP. Note that the annotations are not pixel-perfect.

## Databases
This project utilizes images from several sources:
- **Hawaii**: Coastal images of Hawaii taken across different years.
- **Google Earth**: A global collection of coastal images.
- **New Data**: Images from Douarnenez Bay.

## Reference Work
This project builds on concepts from Conor O'Sullivan's 2022 work on edge detection. If you use parts or the entirety of this code for academic or research purposes, please cite the following paper: [Link to the paper](https://arxiv.org/abs/2405.11494).

### SWED Dataset
We use the Sentinel-2 Water Edges Dataset (SWED) from the UK Hydrographic Office, available under the Geospatial Commission Data Exploration license. More details can be found [here](https://openmldata.ukho.gov.uk/#:~:text=The%20Sentinel%2D2%20Water%20Edges,required%20for%20the%20segmentation%20mask).
