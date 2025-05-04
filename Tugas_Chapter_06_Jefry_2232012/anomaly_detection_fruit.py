import cv2
import numpy as np

def load_and_preprocess(image):
    """
    Preprocess an image by resizing it to a standard size.
    Assumes the image is already in grayscale.
    """
    if image is None:
        raise ValueError("Provided image is None.")
    image = cv2.resize(image, (800, 600))  # Resize for consistency
    return image

def align_images(reference, test):
    """
    Align the test image to the reference image using feature matching.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(reference, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    if des1 is None or des2 is None:
        print("No descriptors found. Skipping alignment.")
        return test

    # Use KNN matching with BFMatcher for better flexibility
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches found. Skipping alignment.")
        return test

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is not None:
        # Warp test image to align with reference
        aligned_test = cv2.warpPerspective(test, M, (reference.shape[1], reference.shape[0]))
        return aligned_test
    else:
        print("Homography could not be computed. Skipping alignment.")
        return test

def detect_anomalies(reference, test):
    """
    Detect anomalies by finding differences between the reference and test images.
    """
    # Compute absolute difference
    diff = cv2.absdiff(reference, test)

    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of the anomalies
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, processed

def detect_busa(test_image):
    """
    Detect busa (foam) based on color thresholding.
    Assume busa is lighter in color (white/cream).
    """
    hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

    # Define range for detecting busa (white/cream areas)
    lower_busa = np.array([0, 0, 200])
    upper_busa = np.array([180, 30, 255])

    # Threshold the image to extract busa-like areas
    mask = cv2.inRange(hsv, lower_busa, upper_busa)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the busa areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def annotate_anomalies(image, contours, color=(0, 0, 255)):
    """
    Draw bounding boxes around detected anomalies.
    color: tuple, BGR color for the bounding boxes (default is red)
    """
    # Check if the image is already in BGR format
    if len(image.shape) == 2:  # Grayscale image
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image  # Image is already in BGR format
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    
    return annotated

def main():
    # Path to the reference image
    reference_path = 'dataset/ripe/apple1.jpg'
    # Path to the test video
    test_video_path = 'dataset/reject/apple1.mp4'

    # Load and preprocess the reference image
    reference_image = load_and_preprocess(cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE))

    # Initialize video capture
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video at path '{test_video_path}'.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Preprocess the frame
        preprocessed_frame = load_and_preprocess(gray_frame)

        # Align the test frame with the reference image
        aligned_test_image = align_images(reference_image, preprocessed_frame)

        # Detect anomalies
        contours, anomaly_mask = detect_anomalies(reference_image, aligned_test_image)

        # Detect busa in the frame
        busa_contours, busa_mask = detect_busa(frame)

        # Annotate anomalies and busa on the test image
        annotated_image = annotate_anomalies(aligned_test_image, contours)
        annotated_image_with_busa = annotate_anomalies(annotated_image, busa_contours, color=(0, 255, 0))  # Green for busa

        # Display the masks and annotated image
        cv2.imshow('Anomaly Mask', anomaly_mask)
        cv2.imshow('Annotated Anomalies & Busa', annotated_image_with_busa)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video processing interrupted by user.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
