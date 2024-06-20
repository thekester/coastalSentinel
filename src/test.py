import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from osgeo import gdal

import os
import glob
import random

from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import rmse, psnr, uqi, ergas, scc, rase, sam, vifp

# Loading metric data
df_metrics = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_all.csv", index_col=0)

# Checking numeric columns for aggregation
numeric_columns = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_columns)

# Calculating mean by method and band
mean_all = df_metrics.groupby(["method", "band"], as_index=False)[numeric_columns].mean()
print("Mean calculated.")

# Creating DataFrame for aggregated statistics
all_stats = mean_all[mean_all["method"] == "canny"][['band', 'psnr', 'ssim']]
all_stats.columns = ['band', 'canny_psnr_mn', 'canny_ssim_mn']

print("Initial DataFrame all_stats:\n", all_stats.head())

# Adding mean values for each method
for method in ["sobel", "scharr", "prewitt"]:
    method_stats = mean_all[mean_all["method"] == method]
    all_stats[method + "_psnr_mn"] = method_stats["psnr"].values
    all_stats[method + "_ssim_mn"] = method_stats["ssim"].values

print("Mean values added to all_stats.")

# Calculating standard deviation by method and band
std_all = df_metrics.groupby(["method", "band"], as_index=False)[numeric_columns].std()
print("Standard deviation calculated.")

# Debugging: Print the standard deviation data for 'canny' method
print("Standard deviation for 'canny' method before filling NaNs:\n", std_all[std_all["method"] == "canny"])

# Adding standard deviation values for each method
for method in ["canny", "sobel", "scharr", "prewitt"]:
    method_stats = std_all[std_all["method"] == method]
    all_stats[method + "_psnr_std"] = method_stats["psnr"].values
    all_stats[method + "_ssim_std"] = method_stats["ssim"].values

# Ensure no NaN values in the standard deviation data
all_stats = all_stats.fillna(0)
print("Standard deviation values added to all_stats.")
print("All_stats DataFrame:\n", all_stats)

# Saving to clipboard and displaying DataFrame
all_stats.to_clipboard()
print("DataFrame saved to clipboard:", all_stats.head())

# NEXT PART TRIAL

# Load your data
df_all = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_all.csv", index_col=0)

print(df_all)

# Define numeric columns
numeric_columns = ['psnr', 'ssim']  # Add other numeric columns if needed

# Check for non-numeric values in numeric columns
for col in numeric_columns:
    non_numeric_rows = df_all[pd.to_numeric(df_all[col], errors='coerce').isnull()]
    if not non_numeric_rows.empty:
        print(f"Non-numeric values in {col}:")
        print(non_numeric_rows)

# Convert numeric columns to float, coercing errors to NaN
df_all[numeric_columns] = df_all[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in numeric columns
df_all = df_all.dropna(subset=numeric_columns)

print("Drop rows with NaN values in numeric columns")
print(df_all)

# Debugging: Check the data for 'canny' method
print("Data for 'canny' method:\n", df_all[df_all['method'] == 'canny'])

# Now perform your groupby operations for 'canny' method
# Filter out non-numeric columns
numeric_columns = df_all.select_dtypes(include=[np.number]).columns

# Group by 'method' and 'band' and calculate mean for numeric columns
mean_all = df_all[df_all.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].mean()

print("mean_all")
print(mean_all)
sd_all = df_all[df_all.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].std()
print("sd_all")
print(sd_all)

# Ensure no NaN values in the standard deviation data
sd_all = sd_all.fillna(0)
print("new sd_all")
print(sd_all)

# Load data from Metrics_blur.csv and Metrics_morph.csv if needed
df_blur = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_blur.csv", index_col=0)
df_morph = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_morph.csv", index_col=0)

# Calculate mean and standard deviation for 'canny' method grouped by 'method' and 'band' for df_blur and df_morph
mean_blur = df_blur[df_blur.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].mean()
sd_blur = df_blur[df_blur.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].std()

mean_morph = df_morph[df_morph.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].mean()
sd_morph = df_morph[df_morph.method == 'canny'].groupby(["method", "band"], as_index=False)[numeric_columns].std()

# Ensure no NaN values in the standard deviation data for blur and morph
sd_blur = sd_blur.fillna(0)
sd_morph = sd_morph.fillna(0)

# Define bands and corresponding channel names
bands = mean_all['band']
channels_ = ['Coastal \nAerosol', 'Blue', 'Green',
             'Red', 'Red Edge 1', 'Red Edge 2',
             'Red Edge 3', 'NIR', 'Red Edge 4',
             'Water \nVapour', 'SWIR 1', 'SWIR 2']

#Part 3
#Fig. 10. Average PSNR for canny edge detection using different noise reduction methods

# Determin impact of post processing
df_hist = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_hist.csv",index_col=0) #no noise reduction
df_all = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_all.csv",index_col=0) #blur
df_morph = pd.read_csv("SWED-edge-detection-main/src/metrics/Metrics_morph.csv",index_col=0) #morph

numeric_columns = df_all.select_dtypes(include=[np.number]).columns
numeric_columns_hist = df_hist.select_dtypes(include=[np.number]).columns
numeric_columns_morph = df_morph.select_dtypes(include=[np.number]).columns


mean_none = df_hist[df_hist.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns_hist].mean()
sd_none = df_hist[df_hist.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns_hist].std()

mean_blur = df_all[df_all.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns].mean()
sd_blur = df_all[df_all.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns].std()

mean_morph = df_morph[df_morph.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns_morph].mean()
sd_morph = df_morph[df_morph.method=='canny'].groupby(["method","band"],as_index=False)[numeric_columns_morph].std()

bands = mean_none['band']

