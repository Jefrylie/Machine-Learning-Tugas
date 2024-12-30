import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_and_preprocess(image_path):
    """
    Load an image in RGB and resize it to a standard size.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.resize(image, (800, 600))  # Resize for consistency
    return image

def detect_anomalies(image):
    """
    Detect anomalies by applying K-Means clustering to color features.
    """
    # Convert image to 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)  # Change the number of clusters based on dataset
    kmeans.fit(pixels)

    # Get the labels
    labels = kmeans.labels_

    # Reshape labels to match image dimensions
    label_image = labels.reshape(image.shape[0], image.shape[1])

    # Assuming the cluster with the least count is the anomaly
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    anomaly_cluster = min(cluster_counts, key=cluster_counts.get)

    # Create a mask for the anomaly
    anomaly_mask = np.zeros_like(image)  # Initialize mask with zeros (black)
    anomaly_mask[label_image == anomaly_cluster] = [0, 0, 255]  # Red color for anomalies

    return anomaly_mask

def main():
    # Path to the test image.
    test_path = 'dataset/reject/Tomat01.jpg'

    # Load and preprocess image.
    test_image = load_and_preprocess(test_path)

    # Detect anomalies.
    anomaly_mask = detect_anomalies(test_image)

    # Display results.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Test Image')
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Anomaly Mask')
    plt.imshow(cv2.cvtColor(anomaly_mask, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
