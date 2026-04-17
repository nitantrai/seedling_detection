import pandas as pd

df = pd.read_csv("outputs/results.csv")

total_images = len(df)
total_detections = df["detections"].sum()
avg_detections = df["detections"].mean()

print("Total images processed:", total_images)
print("Total detections:", total_detections)
print("Average detections per image:", round(avg_detections, 2))

summary = pd.DataFrame({
    "total_images": [total_images],
    "total_detections": [total_detections],
    "avg_detections_per_image": [round(avg_detections, 2)]
})

summary.to_csv("outputs/summary.csv", index=False)
print("Summary saved to outputs/summary.csv")
