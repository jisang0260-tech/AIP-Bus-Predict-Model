# AIP BUS

Put a bus video in `data/videos`, run the analyzer, and it will:

1. extract frames into `data/frames/<video_name>`
2. track buses with `yolov8x.pt`
3. save CSV features into `outputs/<video_name>_vehicle_counts.csv`

## Folder Structure

```text
AIP BUS/
  bus_yolo_analyzer.py
  requirements.txt
  data/
    videos/
      bus.mp4
    frames/
      bus/
        frame_000000.jpg
        frame_000001.jpg
  outputs/
    bus_vehicle_counts.csv
```

`data/frames` and `outputs` are generated automatically. You only need to put
the source video in `data/videos`.

## Run From Video

```powershell
.\setup_env.ps1

.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py
```

For a quick test before running the full video:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py --max-frames 3
```

If there is more than one video, choose one:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --video ".\data\videos\bus.mp4"
```

By default, the script extracts 1 frame per second. To extract more frames:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --video ".\data\videos\bus.mp4" `
  --extract-fps 2
```

To force frame extraction again:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --video ".\data\videos\bus.mp4" `
  --overwrite-frames
```

## Run From Existing Frames

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --image-dir ".\data\frames\bus" `
  --output-csv ".\outputs\bus_vehicle_counts.csv"
```

## Predict Next Departure

After `bus_yolo_analyzer.py` creates a CSV, run:

```powershell
.\.venv\Scripts\python.exe .\bus_departure_predictor.py
```

Or choose a CSV directly:

```powershell
.\.venv\Scripts\python.exe .\bus_departure_predictor.py `
  --csv ".\outputs\bus_vehicle_counts.csv"
```

The predictor treats `exited_bus_count > 0` as a departure event. If that column
is missing, it falls back to `bus_count_diff < 0` or tracker ID changes. The
output is saved as `outputs/<csv_name>_departure_probability.csv`.

## Main CSV Columns

- `bus_count_inside`: Tracked bus count inside the ROI.
- `bus_ids_inside`: Current tracker IDs.
- `remaining_time`: Same value as `total_waiting_time`, kept for the next model.
- `total_waiting_time`: Sum of waiting time for buses currently inside the ROI.
- `avg_waiting_time`: Average waiting time for buses currently inside the ROI.
- `max_waiting_time`: Longest waiting time among buses currently inside the ROI.
- `new_bus_count`, `exited_bus_count`: Buses that entered or exited on this frame.

The old notebook's `remaining_time` column represented time until the next scheduled
bus arrival. In this version, `remaining_time` means the total waiting time of
buses currently inside the ROI.
