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

## Gate ROI Direction Setup

Use this mode to draw the two entrance/exit gate areas directly on a video
frame. For each gate:

1. drag a rectangle around the gate ROI
2. drag an arrow in the bus OUT direction
3. press Enter after both gates are set

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --configure-gates `
  --gate-count 2 `
  --gate-config ".\configs\gate_rois.json"
```

The config is saved in original-frame coordinates:

```text
configs/gate_rois.json
configs/gate_rois_preview.jpg
```

The same gate ROI is used for both entering and exiting. Later in/out event
logic should treat movement along the saved arrow as OUT and movement in the
opposite direction as IN.

When a gate config is present, the analyzer now resets:

- `seconds_since_last_new_bus` when a bus moves through a gate ROI in the IN direction
- `exited_bus_count` and `seconds_since_last_out_bus` when a bus moves through a gate ROI in the OUT direction

## Debug Preview

Use this when you want to visually inspect:

- YOLO raw tracker behavior
- logical ID stitching
- held tracks that survive missing frames
- gate ROI overlays
- IN / OUT events on the exact frame they fire

Example: generate only the debug preview mp4.

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --image-dir ".\data\frames\20260519_133357" `
  --output-csv ".\outputs\20260519_133357_vehicle_counts.csv" `
  --gate-config ".\configs\gate_rois.json" `
  --debug-preview
```

The default preview output path is:

```text
outputs/debug_preview/<csv_stem>/preview.mp4
```

To save only annotated jpg frames and skip mp4 creation:

```powershell
.\.venv\Scripts\python.exe .\bus_yolo_analyzer.py `
  --image-dir ".\data\frames\20260519_133357" `
  --skip-frames 721 `
  --max-frames 41 `
  --output-csv ".\outputs\debug_12m22s.csv" `
  --gate-config ".\configs\gate_rois.json" `
  --debug-preview-frames-only
```

That command is useful when you want to inspect only one label window, for example
`00:12:22 +/- 20 sec`.

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
- `seconds_since_last_new_bus`: Seconds since the most recent gate IN event.
- `exited_bus_count`: Seconds since the most recent gate OUT event.
- `seconds_since_last_out_bus`: Same OUT elapsed time, kept as an explicit column.
- `gate_in_event_count`, `gate_out_event_count`: Gate crossing events detected on the current frame.
- `gate_in_event_ids`, `gate_out_event_ids`: Logical IDs that triggered the gate event on the current frame.
- `tracker_exited_bus_count`, `tracker_exited_bus_ids`: Logical tracks removed because they exceeded `max_missing_frames`.
- `avg_waiting_time`: Average waiting time for buses currently inside the ROI.
- `max_waiting_time`: Longest waiting time among buses currently inside the ROI.
- `new_bus_count`, `new_bus_ids`: Newly created logical tracks on the current frame.

The old notebook's `remaining_time` column represented time until the next scheduled
bus arrival. In this version, `remaining_time` means the total waiting time of
buses currently inside the ROI.
