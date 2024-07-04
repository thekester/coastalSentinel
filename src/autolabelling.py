import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tqdm import tqdm
import tensorflow as tf
import threading
import tensorflow.keras.backend as K

# Define the weighted cross entropy function
def weighted_cross_entropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

# Define the combined loss function
def combined_loss():
    def loss(y_true, y_pred):
        # Use the weighted cross entropy loss function
        weights = [0.1, 0.9]  # Example weights
        weighted_loss = weighted_cross_entropy(weights)

        # Calculate the main loss
        main_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_labels = tf.cast(y_pred_labels, tf.float32)  # Ensure y_pred_labels are float32

        print("Debugging combined_loss:")
        print("y_pred_labels dtype:", y_pred_labels.dtype)
        print("y_pred_labels shape:", y_pred_labels.shape)

        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        for filename in manual_label_filenames:
            manual_label_path = os.path.join(manual_label_dir, filename)
            manual_image = cv2.imread(manual_label_path)
            manual_image = cv2.resize(manual_image, (img_width, img_height))
            manual_image = manual_image.astype(np.float32) / 255.0

            # Initialize the manual mask with -1 for undefined pixels
            manual_mask = np.full((img_height, img_width), -1, dtype=np.float32)
            manual_mask[np.all(manual_image == [0, 0, 255], axis=-1)] = 0.0  # Mainland
            manual_mask[np.all(manual_image == [255, 0, 0], axis=-1)] = 1.0  # Water
            manual_mask = tf.convert_to_tensor(manual_mask, dtype=tf.float32)  # Ensure manual_mask is float32

            print("Before correction:")
            print("manual_mask dtype:", manual_mask.dtype)
            print("manual_mask shape:", manual_mask.shape)

            # Ensure the shapes are compatible
            manual_mask = tf.expand_dims(manual_mask, axis=0)  # Add batch dimension
            manual_mask = tf.expand_dims(manual_mask, axis=-1)  # Add channel dimension
            manual_mask = tf.image.resize_with_crop_or_pad(manual_mask, y_pred_labels.shape[1], y_pred_labels.shape[2])
            manual_mask = tf.squeeze(manual_mask, axis=-1)  # Remove channel dimension

            y_pred_labels_squeezed = tf.squeeze(y_pred_labels, axis=0)  # Remove batch dimension

            print("After correction:")
            print("manual_mask shape after resize_with_crop_or_pad and squeeze:", manual_mask.shape)
            print("y_pred_labels_squeezed shape after squeeze:", y_pred_labels_squeezed.shape)

            # Add explanation for adding a dimension
            print("Adding a channel dimension to y_pred_labels_squeezed to match manual_mask dimensions for loss calculation.")

            # Ensure y_pred_labels_squeezed has the right shape for loss calculation
            y_pred_labels_squeezed = tf.expand_dims(y_pred_labels_squeezed, axis=-1)  # Add channel dimension

            print("y_pred_labels_squeezed:", y_pred_labels_squeezed)

            # Ensure that all values in manual_mask are valid indices (0 or 1)
            valid_mask = tf.not_equal(manual_mask, -1.0)
            manual_mask = tf.boolean_mask(manual_mask, valid_mask)
            y_pred_labels_squeezed = tf.boolean_mask(y_pred_labels_squeezed, valid_mask)
            
            print("manual_mask (after filtering undefined pixels):", manual_mask)
            print("manual_mask type:", manual_mask.dtype)
            print("manual_mask shape:", manual_mask.shape)
            print("y_pred_labels_squeezed (after filtering undefined pixels):", y_pred_labels_squeezed)
            print("y_pred_labels_squeezed type:", y_pred_labels_squeezed.dtype)
            print("y_pred_labels_squeezed shape:", y_pred_labels_squeezed.shape)

            # Calculate manual_loss using Mean Squared Error
            manual_loss = mse_loss_fn(manual_mask, y_pred_labels_squeezed)
            print("manual_loss:", manual_loss)
            print("manual_loss type:", type(manual_loss))

            main_loss += manual_loss

        return main_loss + weighted_loss(y_true, y_pred)
    
    return loss

# Define paths to the directories
img_height, img_width = 150, 150
batch_size = 32
epochs = 20
base_dir = os.path.abspath('classification')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
model_path = os.path.abspath('best_segmentation_model.keras')
manual_label_dir = os.path.join(base_dir, 'manual_label/train/both')
manual_label_filenames = [f for f in os.listdir(manual_label_dir) if os.path.isfile(os.path.join(manual_label_dir, f))]

print(f"Base directory: {base_dir}")
print(f"Train directory: {train_dir}")
print(f"Validation directory: {validation_dir}")
print(f"Model path: {model_path}")
print(f"Manual label directory: {manual_label_dir}")

def load_images_from_directory(directory: str, label: int, img_height: int = 150, img_width: int = 150):
    images = []
    labels = []
    filenames = []
    for filename in tqdm(os.listdir(directory), desc=f"Loading {label} images"):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    print(f"Loaded {len(images)} images from {directory}")
    return np.array(images), np.array(labels), filenames

class ImageDisplayCallback(Callback):
    def __init__(self, test_data, filenames, manual_label_filenames, manual_label_dir, img_height, img_width):
        self.test_data = test_data
        self.filenames = filenames
        self.manual_label_filenames = manual_label_filenames
        self.manual_label_dir = manual_label_dir
        self.img_height = img_height
        self.img_width = img_width

    def on_epoch_end(self, epoch, logs=None):
        X_test = self.test_data
        filenames = self.filenames
        indices = np.random.choice(len(X_test), 5, replace=False)
        predictions = self.model.predict(X_test)
        plt.figure(figsize=(20, 9))  # Increase figure height for better spacing
        for i, idx in enumerate(indices):  # Displaying 5 random examples
            ax = plt.subplot(3, 5, i + 1)
            plt.imshow(X_test[idx])
            plt.title(f'Original\n{filenames[idx]}')
            ax.axis('off')
            
            ax = plt.subplot(3, 5, i + 6)
            plt.imshow(np.argmax(predictions[idx], axis=-1), cmap='jet')
            plt.title(f'Predicted\n{filenames[idx]}')
            ax.axis('off')

            if filenames[idx] in self.manual_label_filenames:
                manual_label_path = os.path.join(self.manual_label_dir, filenames[idx])
                manual_label = cv2.imread(manual_label_path)
                manual_label = cv2.resize(manual_label, (self.img_width, self.img_height))
                ax = plt.subplot(3, 5, i + 11)
                plt.imshow(manual_label)
                plt.title(f'Manual Label\n{filenames[idx]}')
                ax.axis('off')

        plt.suptitle(f'Epoch {epoch + 1}')
        plt.tight_layout()
        plt.show()

def get_user_input(prompt, timeout=10, default="yes"):
    def user_input_thread(prompt, result):
        result.append(input(prompt))

    result = []
    thread = threading.Thread(target=user_input_thread, args=(prompt, result))
    thread.start()
    thread.join(timeout)
    if not result:
        return default
    return result[0]

def train(img_height: int = 150, img_width: int = 150, batch_size: int = 32, epochs: int = 20, base_dir: str = 'classification', model_path: str = 'best_segmentation_model.keras', continue_training: bool = False):
    base_dir = os.path.abspath(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    model_path = os.path.abspath(model_path)
    manual_label_dir = os.path.join(base_dir, 'manual_label/train/both')

    print("Starting data loading...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    train_mainland_dir = os.path.join(train_dir, 'mainland')
    train_water_dir = os.path.join(train_dir, 'water')
    train_both_dir = os.path.join(train_dir, 'both')

    mainland_images, mainland_labels, _ = load_images_from_directory(train_mainland_dir, label=0, img_height=img_height, img_width=img_width)
    water_images, water_labels, _ = load_images_from_directory(train_water_dir, label=1, img_height=img_height, img_width=img_width)
    both_images, _, both_filenames = load_images_from_directory(train_both_dir, label=2, img_height=img_height, img_width=img_width)

    manual_label_filenames = [f for f in os.listdir(manual_label_dir) if os.path.isfile(os.path.join(manual_label_dir, f))]

    X = np.concatenate((mainland_images, water_images), axis=0)
    y = np.concatenate((mainland_labels, water_labels), axis=0)

    X = X.astype(np.float32)
    both_images = both_images.astype(np.float32)

    X /= 255.0
    both_images /= 255.0

    def create_pixelwise_labels(images, label):
        labels = np.ones((images.shape[0], img_height, img_width), dtype=np.float32) * label
        return labels

    mainland_pixelwise_labels = create_pixelwise_labels(mainland_images, 0.0)
    water_pixelwise_labels = create_pixelwise_labels(water_images, 1.0)

    y_pixelwise = np.concatenate((mainland_pixelwise_labels, water_pixelwise_labels), axis=0)

    print("Unique values in mainland_pixelwise_labels:", np.unique(mainland_pixelwise_labels))
    print("Unique values in water_pixelwise_labels:", np.unique(water_pixelwise_labels))

    X_train, X_val, y_train, y_val = train_test_split(X, y_pixelwise, test_size=0.2, random_state=42)

    input_img = Input(shape=(img_height, img_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    print("Shape after Conv2D(64):", x.shape)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Shape after MaxPooling2D:", x.shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    print("Shape after Conv2D(32):", x.shape)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    print("Shape after encoding MaxPooling2D:", encoded.shape)

    x = UpSampling2D((2, 2))(encoded)
    print("Shape after UpSampling2D:", x.shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    print("Shape after Conv2D(32) in decoder:", x.shape)
    x = UpSampling2D((2, 2))(x)
    print("Shape after UpSampling2D:", x.shape)
    x = Cropping2D(((1, 1), (1, 1)))(x) 
    print("Shape after Cropping2D:", x.shape)
    decoded = Conv2D(2, (3, 3), activation='softmax', padding='same')(x)
    print("Shape after final Conv2D:", decoded.shape)

    segmentation_model = Model(input_img, decoded)
    segmentation_model.compile(optimizer='adam', loss=combined_loss(), metrics=['accuracy'])

    # Check if the model already exists and prompt the user
    if os.path.exists(model_path):
        continue_training = input("A model already exists. Do you want to continue training it? (yes/no): ")
        if continue_training.lower() == 'yes':
            # Load the model with custom loss function, using safe_mode=False to allow lambda deserialization
            segmentation_model = load_model(model_path, custom_objects={'loss': combined_loss()}, safe_mode=False)  # Use safe_mode=False
        elif continue_training.lower() == 'no':
            print("Training a new model.")
        else:
            print("Invalid input. Training a new model.")

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5),
        ModelCheckpoint(model_path, save_best_only=True),
        ImageDisplayCallback(both_images, both_filenames, manual_label_filenames, manual_label_dir, img_height, img_width)
    ]

    print("Starting training...")
    history = segmentation_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    segmentation_model.load_weights(model_path)

    both_predictions = segmentation_model.predict(both_images)

    both_masks = np.argmax(both_predictions, axis=-1)

    print("Shape of both_images:", both_images.shape)
    print("Shape of both_predictions:", both_predictions.shape)
    print("Shape of both_masks:", both_masks.shape)
    print("Unique values in both_masks:", np.unique(both_masks))

    def visualize_segmentation(images, masks, filenames, manual_label_filenames, manual_label_dir, n=5):
        indices = np.random.choice(len(images), n, replace=False)
        plt.figure(figsize=(20, 9))
        for i, idx in enumerate(indices):
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(images[idx])
            plt.title(f'Original\n{filenames[idx]}')
            plt.axis('off')

            ax = plt.subplot(3, n, i + 1 + n)
            mask = np.zeros((images[idx].shape[0], images[idx].shape[1], 3), dtype=np.uint8)
            for j in range(images[idx].shape[0]):
                for k in range(images[idx].shape[1]):
                    if masks[idx][j, k] == 0:
                        mask[j, k] = [128, 0, 128]  # Mainland in purple
                    elif masks[idx][j, k] == 1:
                        mask[j, k] = [255, 255, 0]  # Water in yellow
            plt.imshow(mask)
            plt.title(f'Segmented\n{filenames[idx]}')
            plt.axis('off')

            if filenames[idx] in manual_label_filenames:
                manual_label_path = os.path.join(manual_label_dir, filenames[idx])
                manual_label = cv2.imread(manual_label_path)
                manual_label = cv2.resize(manual_label, (img_width, img_height))
                manual_label = cv2.cvtColor(manual_label, cv2.COLOR_BGR2RGB)  # Convert to RGB
                manual_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                manual_mask[np.all(manual_label == [0, 0, 255], axis=-1)] = [128, 0, 128]  # Mainland in purple
                manual_mask[np.all(manual_label == [255, 0, 0], axis=-1)] = [255, 255, 0]  # Water in yellow
                ax = plt.subplot(3, n, i + 1 + 2 * n)
                plt.imshow(manual_mask)
                plt.title(f'Manual Label\n{filenames[idx]}')
                plt.axis('off')

        plt.legend(['Mainland in purple', 'Water in yellow'], loc='upper left')
        plt.tight_layout()
        plt.show()

    visualize_segmentation(both_images, both_masks, both_filenames, manual_label_filenames, manual_label_dir)

    def plot_training_history(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.show()

    plot_training_history(history)

if __name__ == "__main__":
    response = get_user_input("Do you want to start training? (yes/no): ")
    if response.lower() == "yes":
        train()
    else:
        print("Training aborted.")
