from ultralytics import YOLO
import cv2

# Load model (change path if needed)
model = YOLO("models/best.pt")  # or your trained model

# Load image
image_path = "data/sample.jpg"
img = cv2.imread(image_path)

# Run prediction
results = model(image_path)

# Extract detections
boxes = results[0].boxes

# Count detections
num_detections = len(boxes)

print(f"Detected objects: {num_detections}")

# Save output image
output_path = "outputs/result.jpg"
results[0].save(filename=output_path)

# Save results to CSV
data = {
    "image": [image_path],
    "detections": [num_detections]
}

df = pd.DataFrame(data)
df.to_csv("outputs/results.csv", index=False)

print("Results saved to outputs/results.csv")
