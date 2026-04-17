import pandas as pd

df = pd.read_csv("outputs/results.csv")

total_images = len(df)
total_detections = df["detections"].sum()
avg_detections = df["detections"].mean()
best_image = df.loc[df["detections"].idxmax(), "image"]

report = f"""
UAV Seedling Detection Report
-----------------------------
Total images processed: {total_images}
Total detections: {total_detections}
Average detections per image: {avg_detections:.2f}
Highest-detection image: {best_image}
"""

with open("outputs/report.txt", "w") as f:
    f.write(report)

print(report)
print("Report saved to outputs/report.txt")
