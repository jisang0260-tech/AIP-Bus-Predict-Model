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

## Tracker Stability

The analyzer now uses `trackers/bus_botsort.yaml` by default and adds a logical
ID layer on top of YOLO's raw tracker IDs. If YOLO briefly misses a bus or gives
the same bus a new raw ID, the script reconnects it using bounding-box overlap
and center-position similarity.

Useful tuning options:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --max-missing-frames 45 `
  --stitch-iou-threshold 0.2 `
  --stitch-center-threshold 0.45
```

For this fixed camera, set `--roi x1,y1,x2,y2` to the same exit/driveway area
after you confirm the correct coordinates. If `--roi` is omitted, the full frame
is analyzed.

## Predict Next Departure

## Manual Label Training

Create a manual departure label CSV after watching the video:

```csv
event_id,event_time_hhmmss,note
1,00:01:35,left exit
2,00:03:30,left exit
3,00:06:28,left exit
```

Save it under `data/labels`, for example:

```text
data/labels/20260520_145353_1_departures.csv
```

Build a training CSV by combining YOLO features with your manual labels:

```powershell
.\.venv\Scripts\python.exe .\build_training_labels.py `
  --features ".\outputs\20260520_145353_1_vehicle_counts.csv" `
  --labels ".\data\labels\20260520_145353_1_departures.csv" `
  --output ".\outputs\training\20260520_145353_1_training.csv"
```

Train both RandomForest models:

```powershell
.\.venv\Scripts\python.exe .\train_departure_model.py `
  --training-dir ".\outputs\training" `
  --model-output ".\models\departure_random_forest.joblib"
```

The model bundle contains:

- `RandomForestRegressor`: predicts how many seconds remain until departure.
- `RandomForestClassifier`: predicts probability by time bucket.

## Predict Next Departure

After `bus_yolo_analyzer.py` creates a CSV and the RandomForest model is trained, run:

```powershell
.\.venv\Scripts\python.exe .\bus_departure_predictor.py
```

Or choose a CSV directly:

```powershell
.\.venv\Scripts\python.exe .\bus_departure_predictor.py `
  --csv ".\outputs\bus_vehicle_counts.csv" `
  --model ".\models\departure_random_forest.joblib"
```

The output includes the regressor's predicted seconds and the classifier's
bucket probabilities. If no trained model exists, the predictor falls back to
the older CSV-history method.

## Main CSV Columns

- `bus_count_inside`: Tracked bus count inside the ROI.
- `bus_ids_inside`: Stable logical tracker IDs.
- `raw_tracker_ids_inside`: Raw YOLO tracker IDs observed on the current frame.
- `raw_id_switch_count`: Raw ID changes that were reconnected to an existing logical ID.
- `recovered_bus_count`: Logical IDs recovered after being missed for one or more frames.
- `remaining_time`: Same value as `total_waiting_time`, kept for the next model.
- `total_waiting_time`: Sum of waiting time for buses currently inside the ROI.
- `avg_waiting_time`: Average waiting time for buses currently inside the ROI.
- `max_waiting_time`: Longest waiting time among buses currently inside the ROI.
- `new_bus_count`, `exited_bus_count`: Buses that entered or exited on this frame.

The old notebook's `remaining_time` column represented time until the next scheduled
bus arrival. In this version, `remaining_time` means the total waiting time of
buses currently inside the ROI.
