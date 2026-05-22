# AIP BUS

This project exports per-frame bus features from image frames with YOLO tracking.

## Run

```powershell
pip install -r requirements.txt

python .\bus_yolo_analyzer.py `
  --image-dir "C:\Users\USER\Downloads\bus_predict_data_file\images" `
  --output-csv ".\vehicle_counts.csv" `
  --model yolov8x.pt `
  --pattern "frame_2_*.jpg" `
  --skip-frames 1200 `
  --start-time 05:08:04
```

## Main CSV columns

- `bus_count_inside`: Tracked bus count inside the ROI.
- `bus_ids_inside`: Current tracker IDs.
- `total_waiting_time`: Sum of waiting time for buses currently inside the ROI.
- `avg_waiting_time`: Average waiting time for buses currently inside the ROI.
- `max_waiting_time`: Longest waiting time among buses currently inside the ROI.
- `new_bus_count`, `exited_bus_count`: Buses that entered or exited on this frame.

The old notebook's `remaining_time` column represented time until the next scheduled
bus arrival. This version removes that column and replaces it with
`total_waiting_time`.
