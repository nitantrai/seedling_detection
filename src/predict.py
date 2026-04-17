from ultralytics import YOLO
import os
import pandas as pd

# Load model
model = YOLO("models/best.pt")

# Folder with images
data_folder = "data"

# Output list
results_list = []

# Loop through all images
for filename in os.listdir(data_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(data_folder, filename)

        print(f"Processing: {filename}")

        # Run prediction
        results = model(image_path)

        # Count detections
        boxes = results[0].boxes

        num_detections = len(boxes)

        if num_detections > 0:
            confidences = boxes.conf.cpu().numpy()
            avg_conf = confidences.mean()
            max_conf = confidences.max()
        else:
            avg_conf = 0
            max_conf = 0

        print(f"Detections: {num_detections}")

        # Save output image
        output_image_path = f"outputs/{filename}"
        results[0].save(filename=output_image_path)

        # Store results
        results_list.append({
            "image": filename,
            "detections": num_detections,
            "avg_confidence": avg_conf,
            "max_confidence": max_conf
        })

# Save all results to CSV
df = pd.DataFrame(results_list)
df.to_csv("outputs/results.csv", index=False)

print("✅ Batch processing complete. Results saved to outputs/results.csv")
