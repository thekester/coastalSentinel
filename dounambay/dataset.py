import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the new image
img_path_2 = '1973.png'
img_2 = mpimg.imread(img_path_2)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Display the image
ax.imshow(img_2)

# Add multiple labels for Mainland
ax.text(100, 100, 'Mainland', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(300, 200, 'Mainland', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(500, 300, 'Mainland', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(700, 100, 'Mainland', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

# Add multiple labels for Sea
ax.text(900, 800, 'Sea', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(1100, 700, 'Sea', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(1000, 900, 'Sea', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
ax.text(800, 1000, 'Sea', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

# Hide the axis
ax.axis('off')

# Save the labeled image
labeled_img_path_2 = 'labeled_image_1973.png'
plt.savefig(labeled_img_path_2, bbox_inches='tight')

# Show the plot
plt.show()

labeled_img_path_2
