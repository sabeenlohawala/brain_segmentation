import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

SLICE_DIR = "/om2/scratch/tmp/sabeen/kwyk_final_uncrop"

with open(os.path.join(SLICE_DIR, "percent_backgrounds.json")) as f:
    percent_backgrounds = json.load(f)

train_percent = dict(filter(lambda x: "train" in x[0], percent_backgrounds.items()))

# Histograms for training slices only
# Extract percentages
percentages = list(train_percent.values())

# Define bins for histogram grouping by 10%
bins = np.arange(0, 1.1, 0.1)

# Create histogram
plt.hist(percentages, bins=bins, edgecolor="black")

# Add labels and title
plt.xlabel("Percentage")
plt.ylabel("Frequency")
plt.title("Histogram of Percentages (Train)")

# Show the plot
# plt.show()

plt.savefig("/om2/user/sabeen/tissue_labeling/train_histogram.png")

# Calculate histogram counts
hist_counts, bin_edges = np.histogram(percentages, bins=bins)

# Print the histogram counts
print("Histogram Counts:")
for count, bin_edge in zip(hist_counts, bin_edges[:-1]):
    print(f"{bin_edge:.1f}-{bin_edge + 0.1:.1f}: {count}")

# Histogram for all slices
# Extract percentages
percentages = list(percent_backgrounds.values())

# Define bins for histogram grouping by 10%
bins = np.arange(0, 1.1, 0.1)

# Create histogram
plt.hist(percentages, bins=bins, edgecolor="black")

# Add labels and title
plt.xlabel("Percentage")
plt.ylabel("Frequency")
plt.title("Histogram of Percentages (All)")

# Show the plot
# plt.show()

plt.savefig("/om2/user/sabeen/tissue_labeling/all_histogram.png")

# Calculate histogram counts
hist_counts, bin_edges = np.histogram(percentages, bins=bins)

# Print the histogram counts
print("Histogram Counts:")
for count, bin_edge in zip(hist_counts, bin_edges[:-1]):
    print(f"{bin_edge:.1f}-{bin_edge + 0.1:.1f}: {count}")
