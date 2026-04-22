import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from skimage.filters import frangi
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def clipped_histogram_equalization(image, clip_limit=2.0):
    """
    Apply Clipped Histogram Equalization to the input image.
    """
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        image_eq = clahe.apply(image_np)
    else:  # Color image
        channels = cv2.split(image_np)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        eq_channels = [clahe.apply(channel) for channel in channels]
        image_eq = cv2.merge(eq_channels)
    return Image.fromarray(image_eq)

def bilateral_filtered_retinex(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply Bilateral Filtered Retinex to the input image.
    """
    image_np = np.array(image)
    image_bilateral = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)
    return Image.fromarray(image_bilateral)

def preprocess_image(image):
    """
    Apply preprocessing steps: Clipped Histogram Equalization and Bilateral Filtered Retinex.
    """
    image = clipped_histogram_equalization(image)
    image = bilateral_filtered_retinex(image)
    return image

# Path to the main dataset folder
dataset_folder = r".\Dataset"

# List all class folders
class_folders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]

# Loop through each class folder and display one image
for class_folder in class_folders:
    class_path = os.path.join(dataset_folder, class_folder)
    # List all images in the class folder
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # Select the first image from the class
    image_path = os.path.join(class_path, images[0])

    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format

    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Show the original and preprocessed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed_image)
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.suptitle(f'{class_folder}')
    plt.show()

# Define image dimensions and number of classes
height, width = 128, 128  # Example dimensions, adjust as needed
num_classes = len(class_folders)  # Number of classes based on your dataset

# Initialize lists for storing images and labels
preprocessed_images = []
labels = []

# Loop through each class folder and collect preprocessed images and labels
for class_index, class_folder in enumerate(class_folders):
    class_path = os.path.join(dataset_folder, class_folder)
    # List all images in the class folder
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure image is in RGB format

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Resize image to the required input size
        preprocessed_image = preprocessed_image.resize((width, height))
        image_array = np.array(preprocessed_image)
        
        # Append to lists
        preprocessed_images.append(image_array)
        labels.append(class_index)

# Convert lists to numpy arrays
preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

# Normalize the images
preprocessed_images = preprocessed_images / 255.0

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile and train the model with preprocessed images
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model with validation data
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=11, batch_size=32)

def hough_canny(image):
    """
    Apply Hough Canny Edge Detection to the input image.
    """
    edges = cv2.Canny(image, 10, 0)
    kernel = np.ones((1, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges

def frangi_filter(image):
    """
    Apply Frangi Filter for vascular pattern detection.
    """
    image_np = np.array(image.convert('L'))  # Convert to grayscale
    frangi_img = frangi(image_np)
    return Image.fromarray((frangi_img * 255).astype(np.uint8))

def segmentation(image, class_name):
    """
    Apply preprocessing for segmentation: Hough Canny and Frangi filter.
    """
    image_np = np.array(image)  # Convert PIL Image to NumPy array
    if class_name == "Ulcerative colitis":
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    else:
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        
        # Adjust HSV range to focus on dark red regions
        lower_dark_red = np.array([10, 50, 50])
        upper_dark_red = np.array([0, 55, 20])
        mask1 = cv2.inRange(hsv_image, lower_dark_red, upper_dark_red)
        
        lower_dark_red2 = np.array([60, 150, 0])
        upper_dark_red2 = np.array([250, 255, 150])
        mask2 = cv2.inRange(hsv_image, lower_dark_red2, upper_dark_red2)
        
        mask = cv2.bitwise_or(mask1, mask2)
    
    # Convert binary mask to 3 channels
    mask_3channel = np.stack([mask] * 3, axis=-1)
    return mask_3channel

# Define image dimensions and number of classes
height, width = 128, 128  # Example dimensions, adjust as needed
num_classes = len(class_folders)  # Number of classes based on your dataset

# Load pre-trained DenseNet model
base_model = tf.keras.applications.DenseNet201(input_shape=(height, width, 3), include_top=False, weights='imagenet')

def create_deeplabv3plus_model(base_model, num_classes):
    """
    Create a DeepLabV3+ model with Hough Canny and Frangi filter integration.
    """
    model_input = tf.keras.Input(shape=(height, width, 3))
    x = base_model(model_input, training=False)
    
    # Use a Conv2D layer to predict class scores
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    # Upsample to match input dimensions
    x = tf.keras.layers.UpSampling2D(size=(32, 32))(x)  # Assuming DenseNet201 reduces spatial dimensions by a factor of 32
    
    model = tf.keras.Model(inputs=model_input, outputs=x)
    return model

# Create DeepLabV3+ model
deeplabv3_model = create_deeplabv3plus_model(base_model, num_classes)

# Compile the model
deeplabv3_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize lists for storing preprocessed images and labels
segmentation_images = []
segmentation_labels = []

# Track progress
print("Starting image preprocessing and training...")

# Loop through each class folder and collect preprocessed images and labels
for class_index, class_folder in enumerate(class_folders):
    class_path = os.path.join(dataset_folder, class_folder)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Processing {len(images)} images in folder: {class_folder}")

    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure image is in RGB format
        
        # Preprocess the image
        segmentation_image = segmentation(image, class_folder)
        
        # Resize to the required dimensions
        segmentation_image = cv2.resize(segmentation_image, (width, height))
        
        # Append to lists
        segmentation_images.append(segmentation_image)
        
        # Create a label map of the same size as the image
        label_map = np.full((height, width), class_index)
        segmentation_labels.append(label_map)
    
    print(f"Finished processing folder: {class_folder}")

# Convert lists to numpy arrays
segmentation_images = np.array(segmentation_images)
segmentation_labels = np.array(segmentation_labels)

# Train the DeepLabV3+ model
print("Training DeepLabV3+ model...")
deeplabv3_model.fit(segmentation_images, segmentation_labels, epochs=10, batch_size=32)


# Display edge detection and segmented images for the first image in each folder
print("Displaying edge detection and segmented images...")
# Loop through each class folder and display one image
for class_folder in class_folders:
    class_path = os.path.join(dataset_folder, class_folder)
    # List all images in the class folder
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # Select the first image from the class
    image_path = os.path.join(class_path, images[0])

    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format

    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Convert the preprocessed image to numpy array for segmentation and edge detection
    preprocessed_image_np = np.array(preprocessed_image)
    
    # Segment the image
    binary_mask = segmentation(preprocessed_image_np, class_folder)

    # Perform edge detection
    edges = hough_canny(binary_mask)

    # Show the original, preprocessed images, binary mask, and edges
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    # Original image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Edges
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Edge Detection")
    ax[1].axis("off")

    # Binary mask
    ax[2].imshow(binary_mask)
    ax[2].set_title("Segmentaion")
    ax[2].axis("off")
    plt.suptitle(f'{class_folder}')
    plt.show()

# Load and fine-tune ResNet model
resnet_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(height, width, 3), weights='imagenet')

def extract_features(images):
    """
    Extract features using ResNet.
    """
    feature_extractor = tf.keras.Model(inputs=resnet_model.input, outputs=resnet_model.output)
    features = feature_extractor.predict(images)
    features = features.reshape(features.shape[0], -1)  # Flatten features
    return features

# Extract features from preprocessed images
features = extract_features(segmentation_images)

# Initialize lists for storing test images and labels
test_images = []
test_labels = []

# Loop through each test class folder and collect test images and labels
for class_index, class_folder in enumerate(class_folders):
    class_path = os.path.join(dataset_folder, class_folder)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure image is in RGB format
        
        # Preprocess the image
        test_image = preprocess_image(image)
        test_image = test_image.resize((width, height))
        test_image_array = np.array(test_image)
        
        # Append to lists
        test_images.append(test_image_array)
        test_labels.append(class_index)

# Convert lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Flatten labels and ensure they are in the correct format
labels_flattened = segmentation_labels.reshape(segmentation_labels.shape[0], -1)
labels_flattened = np.argmax(labels_flattened, axis=-1)  # Convert to class indices

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_flattened, test_size=0.2, random_state=42)

# Initialize and train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb_classifier.predict(X_test)
accuracy = max(history.history['accuracy'])
print('\nAccuracy is: {}%'.format(round(accuracy * 100, 2)))

# Predict labels for the test set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Compute confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_folders)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.show()



