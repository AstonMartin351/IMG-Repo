import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from datetime import datetime
from win32api import GetSystemMetrics  # For screen size
from PIL import ImageGrab  # For screen capture
from concurrent.futures import ThreadPoolExecutor
from collections import Counter # For debugging label distribution

# Define folder path for image processing
folder_path = r"C:\Users\benne\Desktop\Image Classifier_ Data & Code\Data_File_3_Upd_Color_Block_Pics"

# Helper function: Normalize brightness
def normalize_brightness(image_hsv):
    """Normalize brightness using scaling instead of equalization."""
    v_channel = image_hsv[:, :, 2]
    v_normalized = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image_hsv[:, :, 2] = v_normalized
    return image_hsv

def classify_color(image):
    """Classify the color of the image based on HSV ranges."""
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for colors
    color_ranges = {
        "Red": [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],  # Updated ranges for red
        "Blue": [(100, 150, 0), (140, 255, 255)],  # Updated ranges for blue
        "Green": [(40, 50, 50), (80, 255, 255)]  # Updated ranges for green
    }

    for color, ranges in color_ranges.items():
        if color == "Red":
            # Handle two ranges for red
            mask1 = cv2.inRange(hsv_image, np.array(ranges[0]), np.array(ranges[1]))
            mask2 = cv2.inRange(hsv_image, np.array(ranges[2]), np.array(ranges[3]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Single range for other colors
            mask = cv2.inRange(hsv_image, np.array(ranges[0]), np.array(ranges[1]))

        if cv2.countNonZero(mask) > 0:
            return color

    return "Unknown"

def process_single_image(file_path):
    """Process a single image to classify its color."""
    try:
        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)  # Resize image for faster processing
        
        # Classify the color based on updated HSV ranges
        label = classify_color(img_resized)
        print(f"Processed image: {file_path}, Detected Color: {label}")
        
        return img_resized.flatten(), label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def load_and_process_images_sequential(folder_path):
    """Load and process all images in a folder."""
    data, labels = [], []
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f"Files found: {files}")  # Debugging file list

    for file_path in files:
        img_data, label = process_single_image(file_path)
        if img_data is not None:
            data.append(img_data)
            labels.append(label)

    # Debugging label information
    print(f"Label distribution: {Counter(labels)}")
    print(f"Unique labels: {len(set(labels))}")
    print(f"Labels: {set(labels)}")
    
    return np.asarray(data), np.asarray(labels)

# Step 2: Train and evaluate the classifier
def train_and_evaluate_classifier(data, labels):
    """Train and evaluate SVM classifier with class check."""
    # Class check: Ensure at least two distinct classes are present
    if len(set(labels)) < 2:
        print("Error: Not enough distinct classes to train the classifier.")
        print(f"Labels found: {set(labels)}")
        return
    
    # Encode string labels into numerical format
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split into training and testing sets
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
        )
    except ValueError as e:
        print("Error in train-test split:", e)
        return
    
    # Train an SVM with hyperparameter tuning
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train)
    
    # Evaluate performance
    best_estimator = grid_search.best_estimator_
    y_prediction = best_estimator.predict(x_test)
    score = accuracy_score(y_test, y_prediction)
    
    # Print accuracy and confusion matrix
    print(f"{score * 100:.2f}% of samples were correctly classified.")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_prediction)}")


# Step 3: Webcam recording with screen capture
def classify_webcam_frame(frame):
    """Classify a single frame from the webcam feed and detect color blocks."""
    # Convert the frame to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_image = normalize_brightness(hsv_image)

    # Define masks for colors
    red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv_image, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    blue_mask = cv2.inRange(hsv_image, np.array([100, 150, 0]), np.array([140, 255, 255]))
    green_mask = cv2.inRange(hsv_image, np.array([40, 50, 50]), np.array([80, 255, 255]))

    # Detect contours for each color
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label = "None"  # Default label
    for contour in contours_red:
        if cv2.contourArea(contour) > 500:  # Filter small regions
            # Draw a bounding box around the detected object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = "Red"
    
    for contour in contours_blue:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = "Blue"

    for contour in contours_green:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "Green"

    # Display the label
    cv2.putText(frame, f"Detected: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return label

def record_video_with_webcam():
    webcam = cv2.VideoCapture(1)  # Index 1 for default webcam
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture webcam frame.")
            break
        
        # Classify the color of the frame
        label = classify_webcam_frame(frame)
        
        # Show the frame
        cv2.putText(frame, f"Label: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Webcam Color Classification", frame)
        
        # Exit on pressing 'p'
        if cv2.waitKey(1) == ord('p'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    # Load and process images
    data, labels = load_and_process_images_sequential(folder_path)
    if data is not None and labels is not None:
        print(f"Processed {len(data)} images with {len(set(labels))} unique labels.")
    
    # Train and evaluate the classifier
    train_and_evaluate_classifier(data, labels)
    
    # Now, start the webcam and screen capture recording
    
    record_video_with_webcam()
