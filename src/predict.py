from ultralytics import YOLO
import cv2

# Load model (change path if needed)
model = YOLO("models/best.pt")  # or your trained model

# Load image
image_path = "data/sample.jpg"
img = cv2.imread(image_path)

# Run prediction
results = model(image_path)

# Save output image
output_path = "outputs/result.jpg"
results[0].save(filename=output_path)

print("Prediction complete. Saved to:", output_path)
