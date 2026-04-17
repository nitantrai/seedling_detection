## Problem
Manual seedling counting from UAV imagery is time-consuming and difficult to scale.

## Approach
This pipeline uses a YOLO-based deep learning model to detect seedlings from aerial imagery and generate structured outputs for monitoring.

## Tools Used
- Python
- Ultralytics YOLO
- Pandas

## Output Files
- `outputs/results.csv`: per-image detection results
- `outputs/summary.csv`: aggregated summary statistics

## Future Improvements
- Density estimation
- Geospatial outputs
- Dashboard visualization
- Multi-modal fusion with LiDAR
