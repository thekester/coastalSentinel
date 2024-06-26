import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
from osgeo import gdal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Lists all items in the current directory
directories = os.listdir()
print(directories)

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
    # If you then want to go to the 'src' subfolder
    if 'src' in os.listdir():
        os.chdir('src')
        print(f"Path changed to subfolder 'src': {os.getcwd()}")
    else:
        print("'src' subfolder does not exist.")

# Function to load and preprocess images
def load_test(path):
    img = gdal.Open(path).ReadAsArray()
    stack_img = np.stack(img, axis=-1)
    rgb_img = get_rgb(stack_img)
    label_path = path.replace("images", "labels").replace("image", "label")
    label = gdal.Open(label_path).ReadAsArray() if os.path.exists(label_path) else None
    return stack_img, rgb_img, label

def get_rgb(img):
    """Return normalized RGB channels from sentinel image"""
    rgb_img = img[:, :, [3,2,1]]
    rgb_normalize = np.clip(rgb_img/10000, 0, 0.3)/0.3
    return (rgb_normalize * 255).astype(np.uint8)  # Convert to uint8

# Load images and labels
def load_data():
    test_path = glob.glob("../data/SWED/test/images/*")
    print(test_path[0])
    print(len(test_path))

    input_images = []
    rgb_images = []
    labels = []
    image_names = []

    for path in tqdm(test_path, desc="Loading data"):
        try:
            img, rgb_img, label = load_test(path)
            input_images.append(img)
            rgb_images.append(rgb_img)
            labels.append(label)
            image_names.append(os.path.basename(path))  # Store image name
        except Exception as e:
            print(f"Error with image {path}: {e}")

    return rgb_images, labels, image_names

# Function to generate label image where blue water becomes white and land becomes black
def generate_label(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define multiple color ranges for detecting different shades of blue
    lower_blue1 = np.array([90, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    
    lower_blue2 = np.array([90, 50, 50])
    upper_blue2 = np.array([110, 255, 255])
    
    lower_blue3 = np.array([100, 50, 50])
    upper_blue3 = np.array([140, 255, 255])
    
    # Create masks for the different blue ranges
    mask_blue1 = cv2.inRange(hsv_image, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(hsv_image, lower_blue2, upper_blue2)
    mask_blue3 = cv2.inRange(hsv_image, lower_blue3, upper_blue3)
    
    # Combine the masks
    mask_water = cv2.bitwise_or(mask_blue1, mask_blue2)
    mask_water = cv2.bitwise_or(mask_water, mask_blue3)

    # Additional step: Use Canny edge detection to help with water boundary detection
    edges = cv2.Canny(image, 100, 200)
    
    # Combine mask and edges
    mask_combined = cv2.bitwise_or(mask_water, edges)
    
    # Initialize the label image with the same dimensions as the input image
    label_image = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Set water regions to white (255)
    label_image[mask_combined > 0] = 255
    
    return label_image

# Load and preprocess the data
rgb_images, labels, image_names = load_data()

# Create output directory if it doesn't exist
output_dir = "../data/SWED/reference_image_generate/"
os.makedirs(output_dir, exist_ok=True)

# Process each image and save the generated label
for rgb_image, image_name in zip(rgb_images, image_names):
    label_image = generate_label(rgb_image)
    output_path = os.path.join(output_dir, image_name.replace(".png", "_reference_generate.png"))
    cv2.imwrite(output_path, label_image)

    # Display the original and label images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(rgb_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('reference generate Image')
    plt.imshow(label_image, cmap='gray')
    plt.axis('off')

    plt.show()

print("Welcome to coastal sentinel model training...")

# Prepare the data for U-Net training
def prepare_data(images, labels):
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels) / 255.0  # Normalize labels
    labels = labels[..., np.newaxis]  # Add channel dimension to labels
    return images, labels

swed_images, swed_labels = prepare_data(rgb_images, labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(swed_images, swed_labels, test_size=0.2, random_state=42)

# Define the U-Net model
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the U-Net model
model = unet_model()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Save the model
model.save('coastline_segmentation_model.h5')

# Evaluate the model
evaluation = model.evaluate(X_val, y_val)
print(f"Model evaluation: {evaluation}")

print("Welcome to the new challenge: Implement a new dataset and see what the model can do...")

# Load and preprocess a new image for prediction
new_image_path = '/mnt/data/image.png'
new_image = cv2.imread(new_image_path)
new_image_resized = cv2.resize(new_image, (256, 256)) / 255.0

# Predict the label for the new image
predicted_label = model.predict(np.expand_dims(new_image_resized, axis=0))[0]

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('New Image')
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Predicted Label')
plt.imshow(predicted_label[..., 0], cmap='gray')
plt.axis('off')

plt.show()

print("Erm it seems that the model don't appreciate the new image")
print("It's because there are so many changes compare to the SWED databse where raw images is with the same colors")
print("So let's improve our model with the hawai dataset")

# Load images in ../hawai/
# Load annotated images in ../hawai/annotated
# In the annotated image, yellow for water/sea and purple for mainland

def load_hawaii_data(image_path_pattern, label_path_pattern):
    image_paths = glob.glob(image_path_pattern)
    label_paths = glob.glob(label_path_pattern)
    
    images = [cv2.imread(img_path) for img_path in image_paths]
    labels = [cv2.imread(lbl_path) for lbl_path in label_paths]
    
    images = np.array([cv2.resize(img, (256, 256)) for img in images])
    labels = np.array([cv2.resize(lbl, (256, 256)) for lbl in labels])
    
    # Convert yellow to white and purple to black
    yellow = np.array([255, 255, 0])
    purple = np.array([128, 0, 128])
    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])
    
    for i, lbl in enumerate(labels):
        lbl[np.all(lbl == yellow, axis=-1)] = white
        lbl[np.all(lbl == purple, axis=-1)] = black
        labels[i] = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY) / 255.0  # Convert to binary mask and normalize

    labels = labels[..., np.newaxis]  # Add channel dimension to labels
    images = images / 255.0  # Normalize images

    return images, labels

hawaii_images, hawaii_labels = load_hawaii_data("../data/hawai/*.png", "../data/hawai/annotated/*.png")

# Combine SWED and Hawaii datasets
combined_images = np.concatenate((swed_images, hawaii_images), axis=0)
combined_labels = np.concatenate((swed_labels, hawaii_labels), axis=0)

# Split the combined data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(combined_images, combined_labels, test_size=0.2, random_state=42)

# Re-train the U-Net model with the combined dataset
model = unet_model()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Save the improved model
model.save('improved_coastline_segmentation_model.h5')

# Evaluate the improved model
evaluation = model.evaluate(X_val, y_val)
print(f"Improved model evaluation: {evaluation}")

# Predict the label for the new image again
predicted_label_improved = model.predict(np.expand_dims(new_image_resized, axis=0))[0]

# Display the results of the improved model
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('New Image')
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Predicted Label (Improved Model)')
plt.imshow(predicted_label_improved[..., 0], cmap='gray')
plt.axis('off')

plt.show()
