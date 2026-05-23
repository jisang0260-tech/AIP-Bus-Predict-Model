from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_ROI = (0, 380, 1400, 700)
DEFAULT_BUS_CLASSES = (5,)
DEFAULT_VIDEO_DIR = Path("data/videos")
DEFAULT_FRAMES_DIR = Path("data/frames")
DEFAULT_OUTPUT_DIR = Path("outputs")
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class DetectionSummary:
    count: int
    ids: list[int]
    raw_ids: list[int]
    areas: list[float]
    waiting_times: list[float]
    new_ids: list[int]
    exited_ids: list[int]
    recovered_ids: list[int]
    raw_id_switches: int


@dataclass(frozen=True)
class TrackObservation:
    raw_id: int | None
    box: tuple[float, float, float, float]
    area: float
    center_x: float
    center_y: float


@dataclass
class LogicalTrack:
    logical_id: int
    first_seen_time: float
    last_seen_frame: int
    last_box: tuple[float, float, float, float]
    raw_ids: set[int]


def parse_time_to_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("time must be HH:MM or HH:MM:SS")

    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise argparse.ArgumentTypeError("time values are out of range")

    return hour * 3600 + minute * 60 + second


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(round(seconds)) % (24 * 3600)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def parse_int_tuple(value: str, expected_length: int, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from exc

    if len(parsed) != expected_length:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected_length} comma-separated integers"
        )
    return parsed


def parse_classes(value: str) -> list[int]:
    class_ids = parse_int_tuple(value, len(value.split(",")), "--classes")
    return list(class_ids)


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def collect_images(image_dir: Path, pattern: str, skip_frames: int) -> list[Path]:
    image_paths = sorted(image_dir.glob(pattern), key=natural_key)
    if skip_frames:
        image_paths = image_paths[skip_frames:]
    return image_paths


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "video"


def discover_video(video_dir: Path) -> Path:
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        [
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=natural_key,
    )

    if not video_paths:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No video found in {video_dir}. Put one video there or pass --video. "
            f"Supported extensions: {supported}"
        )

    if len(video_paths) > 1:
        choices = "\n".join(f"  - {path}" for path in video_paths)
        raise ValueError(
            "More than one video was found. Choose one with --video:\n" + choices
        )

    return video_paths[0]


def assert_child_path(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {parent_resolved}: {child_resolved}") from exc


def reset_directory(path: Path, allowed_root: Path) -> None:
    assert_child_path(path, allowed_root)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    extract_fps: float,
    overwrite: bool,
) -> tuple[int, float]:
    if extract_fps <= 0:
        raise ValueError("--extract-fps must be greater than zero")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for video input. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("frame_*.jpg"), key=natural_key)
    if existing_frames and not overwrite:
        log(f"Using {len(existing_frames)} existing frames from {output_dir}")
        return len(existing_frames), 1.0 / extract_fps

    reset_directory(output_dir, output_dir.parent)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if source_fps <= 0:
        source_fps = extract_fps

    effective_fps = min(extract_fps, source_fps)
    sample_interval = 1.0 / effective_fps
    next_sample_time = 0.0
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_time = frame_index / source_fps
        if frame_time + (0.5 / source_fps) >= next_sample_time:
            output_path = output_dir / f"frame_{saved_count:06d}.jpg"
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                raise RuntimeError(f"Could not encode frame {frame_index}")
            output_path.write_bytes(encoded.tobytes())
            saved_count += 1
            next_sample_time += sample_interval

        frame_index += 1

    capture.release()

    if saved_count == 0:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    log(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count, sample_interval


def preprocess_image(
    image_path: Path,
    roi: tuple[int, int, int, int] | None,
    scale: float,
    contrast: float,
    sharpness: float,
    median_filter_size: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")

    if roi is not None:
        image = image.crop(roi)

    if scale != 1.0:
        width, height = image.size
        image = image.resize((int(width * scale), int(height * scale)))

    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

    if median_filter_size > 1:
        if median_filter_size % 2 == 0:
            raise ValueError("median_filter_size must be an odd number")
        image = image.filter(ImageFilter.MedianFilter(size=median_filter_size))

    return image


def iter_boxes(result) -> Iterable:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    return boxes


def get_track_id(box) -> int | None:
    if getattr(box, "id", None) is None:
        return None

    track_tensor = box.id
    if track_tensor.numel() == 0:
        return None

    return int(track_tensor.reshape(-1)[0].item())


def get_observations(result) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    for box in iter_boxes(result):
        x1, y1, x2, y2 = (
            float(value.item()) for value in box.xyxy[0]
        )
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        observations.append(
            TrackObservation(
                raw_id=get_track_id(box),
                box=(x1, y1, x2, y2),
                area=area,
                center_x=x1 + width / 2.0,
                center_y=y1 + height / 2.0,
            )
        )
    return observations


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = box_area(left) + box_area(right) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def geometric_match_score(
    observation: TrackObservation,
    track: LogicalTrack,
    iou_threshold: float,
    center_threshold: float,
) -> float | None:
    iou = box_iou(observation.box, track.last_box)
    if iou >= iou_threshold:
        return 2.0 + iou

    track_area = box_area(track.last_box)
    area_ratio = min(observation.area, track_area) / max(observation.area, track_area, 1.0)
    obs_center = (observation.center_x, observation.center_y)
    track_center = box_center(track.last_box)
    distance = math.dist(obs_center, track_center)
    base_size = max(math.sqrt(max(observation.area, track_area, 1.0)), 1.0)
    normalized_distance = distance / base_size

    if normalized_distance <= center_threshold and area_ratio >= 0.45:
        return 1.0 - normalized_distance

    return None


def summarize_detections(
    result,
    current_time: float,
    logical_tracks: dict[int, LogicalTrack],
    raw_to_logical: dict[int, int],
    next_logical_id: int,
    frame_index: int,
    max_missing_frames: int,
    stitch_iou_threshold: float,
    stitch_center_threshold: float,
) -> tuple[DetectionSummary, int]:
    observations = get_observations(result)
    assigned_tracks: set[int] = set()
    assigned_observations: set[int] = set()
    new_ids: list[int] = []
    recovered_ids: list[int] = []
    raw_id_switches = 0

    def assign_observation(
        observation: TrackObservation,
        logical_id: int,
    ) -> None:
        nonlocal raw_id_switches
        track = logical_tracks[logical_id]
        if frame_index - track.last_seen_frame > 1:
            recovered_ids.append(logical_id)

        if observation.raw_id is not None:
            if track.raw_ids and observation.raw_id not in track.raw_ids:
                raw_id_switches += 1
            track.raw_ids.add(observation.raw_id)
            raw_to_logical[observation.raw_id] = logical_id

        track.last_seen_frame = frame_index
        track.last_box = observation.box
        assigned_tracks.add(logical_id)

    for observation_index, observation in enumerate(observations):
        if observation.raw_id is None or observation.raw_id not in raw_to_logical:
            continue

        logical_id = raw_to_logical[observation.raw_id]
        track = logical_tracks.get(logical_id)
        if track is None or logical_id in assigned_tracks:
            continue

        if frame_index - track.last_seen_frame <= max_missing_frames:
            assign_observation(observation, logical_id)
            assigned_observations.add(observation_index)

    for observation_index, observation in enumerate(observations):
        if observation_index in assigned_observations:
            continue

        best_logical_id: int | None = None
        best_score: float | None = None
        for logical_id, track in logical_tracks.items():
            if logical_id in assigned_tracks:
                continue

            if frame_index - track.last_seen_frame > max_missing_frames:
                continue

            score = geometric_match_score(
                observation=observation,
                track=track,
                iou_threshold=stitch_iou_threshold,
                center_threshold=stitch_center_threshold,
            )
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_logical_id = logical_id

        if best_logical_id is not None:
            assign_observation(observation, best_logical_id)
            assigned_observations.add(observation_index)
            continue

        logical_id = next_logical_id
        next_logical_id += 1
        raw_ids = {observation.raw_id} if observation.raw_id is not None else set()
        logical_tracks[logical_id] = LogicalTrack(
            logical_id=logical_id,
            first_seen_time=current_time,
            last_seen_frame=frame_index,
            last_box=observation.box,
            raw_ids=raw_ids,
        )
        if observation.raw_id is not None:
            raw_to_logical[observation.raw_id] = logical_id
        assigned_tracks.add(logical_id)
        assigned_observations.add(observation_index)
        new_ids.append(logical_id)

    stale_ids = sorted(
        logical_id
        for logical_id, track in logical_tracks.items()
        if frame_index - track.last_seen_frame > max_missing_frames
    )
    for logical_id in stale_ids:
        track = logical_tracks.pop(logical_id)
        for raw_id in track.raw_ids:
            if raw_to_logical.get(raw_id) == logical_id:
                raw_to_logical.pop(raw_id, None)

    active_tracks = [
        track
        for track in logical_tracks.values()
        if frame_index - track.last_seen_frame <= max_missing_frames
    ]
    active_tracks = sorted(active_tracks, key=lambda track: track.logical_id)
    areas = [box_area(track.last_box) for track in active_tracks]
    waiting_times = [
        current_time - track.first_seen_time
        for track in active_tracks
    ]
    raw_ids = sorted(
        observation.raw_id
        for observation in observations
        if observation.raw_id is not None
    )

    return (
        DetectionSummary(
            count=len(active_tracks),
            ids=[track.logical_id for track in active_tracks],
            raw_ids=raw_ids,
            areas=areas,
            waiting_times=waiting_times,
            new_ids=sorted(new_ids),
            exited_ids=stale_ids,
            recovered_ids=sorted(set(recovered_ids)),
            raw_id_switches=raw_id_switches,
        ),
        next_logical_id,
    )


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract video frames, track buses with a YOLO x-size model, "
            "and export per-frame CSV features."
        )
    )
    parser.add_argument(
        "--video",
        default=None,
        type=Path,
        help="Video file to convert into frames before analysis.",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        type=Path,
        help="Directory used when --video is omitted. Must contain exactly one video.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        type=Path,
        help="Use an existing frame image directory instead of a video.",
    )
    parser.add_argument(
        "--frames-dir",
        default=DEFAULT_FRAMES_DIR,
        type=Path,
        help="Directory where extracted video frames are stored.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="CSV output path. Defaults to outputs/<video_name>_vehicle_counts.csv.",
    )
    parser.add_argument("--pattern", default="frame_*.jpg")
    parser.add_argument("--skip-frames", default=0, type=int)
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Analyze only the first N frames after skipping. Useful for quick tests.",
    )
    parser.add_argument("--start-time", default="00:00:01", type=parse_time_to_seconds)
    parser.add_argument(
        "--frame-interval-sec",
        default=None,
        type=float,
        help="Seconds between analyzed frames. Defaults to 1 / --extract-fps for video.",
    )
    parser.add_argument(
        "--extract-fps",
        default=1.0,
        type=float,
        help="Frames per second to extract from video. Default is 1 frame per second.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Re-extract video frames even if the frame folder already exists.",
    )
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--tracker", default="trackers/bus_botsort.yaml")
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    parser.add_argument(
        "--classes",
        default=",".join(str(class_id) for class_id in DEFAULT_BUS_CLASSES),
        help="COCO class IDs to track. Bus is 5. Use 5,7 if buses are often read as trucks.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Optional x1,y1,x2,y2 crop. For a fixed camera, set this to the "
            "exit/driveway area once it is confirmed."
        ),
    )
    parser.add_argument("--no-roi", action="store_true")
    parser.add_argument("--scale", default=1.2, type=float)
    parser.add_argument("--contrast", default=1.5, type=float)
    parser.add_argument("--sharpness", default=2.0, type=float)
    parser.add_argument("--median-filter-size", default=3, type=int)
    parser.add_argument(
        "--max-missing-frames",
        default=30,
        type=int,
        help="Keep a logical bus ID alive for this many missed frames.",
    )
    parser.add_argument(
        "--stitch-iou-threshold",
        default=0.25,
        type=float,
        help="Reconnect a changed tracker ID when box IoU is at least this value.",
    )
    parser.add_argument(
        "--stitch-center-threshold",
        default=0.35,
        type=float,
        help="Reconnect a changed tracker ID when normalized center movement is small.",
    )
    parser.add_argument(
        "--progress-every",
        default=10,
        type=int,
        help="Print progress every N frames.",
    )
    return parser


def resolve_source(args: argparse.Namespace) -> tuple[list[Path], Path, float]:
    if args.image_dir is not None:
        image_paths = collect_images(args.image_dir, args.pattern, args.skip_frames)
        output_csv = args.output_csv or DEFAULT_OUTPUT_DIR / "vehicle_counts.csv"
        frame_interval_sec = (
            args.frame_interval_sec if args.frame_interval_sec is not None else 1.0
        )
        return image_paths, output_csv, frame_interval_sec

    video_path = args.video if args.video is not None else discover_video(args.video_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    video_frame_dir = args.frames_dir / safe_stem(video_path)
    _, extracted_interval = extract_frames_from_video(
        video_path=video_path,
        output_dir=video_frame_dir,
        extract_fps=args.extract_fps,
        overwrite=args.overwrite_frames,
    )

    image_paths = collect_images(video_frame_dir, args.pattern, args.skip_frames)
    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else DEFAULT_OUTPUT_DIR / f"{safe_stem(video_path)}_vehicle_counts.csv"
    )
    frame_interval_sec = (
        args.frame_interval_sec
        if args.frame_interval_sec is not None
        else extracted_interval
    )
    return image_paths, output_csv, frame_interval_sec


def main() -> None:
    args = build_parser().parse_args()

    if args.skip_frames < 0:
        raise ValueError("--skip-frames cannot be negative")

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero")

    if args.frame_interval_sec is not None and args.frame_interval_sec <= 0:
        raise ValueError("--frame-interval-sec must be greater than zero")

    if args.extract_fps <= 0:
        raise ValueError("--extract-fps must be greater than zero")

    if args.max_missing_frames < 0:
        raise ValueError("--max-missing-frames cannot be negative")

    if not 0 <= args.stitch_iou_threshold <= 1:
        raise ValueError("--stitch-iou-threshold must be between 0 and 1")

    if args.stitch_center_threshold < 0:
        raise ValueError("--stitch-center-threshold cannot be negative")

    roi = None if args.no_roi or args.roi is None else parse_int_tuple(args.roi, 4, "--roi")
    class_ids = parse_classes(args.classes)
    image_paths, output_csv, frame_interval_sec = resolve_source(args)
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found with pattern {args.pattern}. Check the video or frame folder."
        )

    log("Loading YOLO/torch. First run can take 1-2 minutes on Windows.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    log(f"Loading model: {args.model}")
    model = YOLO(args.model)
    log(
        f"Starting analysis: {len(image_paths)} frames, "
        f"frame interval {frame_interval_sec:.3f}s, classes {class_ids}"
    )
    if args.device is None:
        log("Device: auto. This PC appears to be CPU-only, so YOLOv8x can be slow.")
    else:
        log(f"Device: {args.device}")

    rows: list[dict[str, object]] = []
    logical_tracks: dict[int, LogicalTrack] = {}
    raw_to_logical: dict[int, int] = {}
    next_logical_id = 1
    previous_count = 0
    previous_avg_area = 0.0
    last_new_bus_time: float | None = None

    total_frames = len(image_paths)
    for frame_index, image_path in enumerate(image_paths):
        if (
            args.progress_every > 0
            and (frame_index == 0 or (frame_index + 1) % args.progress_every == 0)
        ):
            log(f"Analyzing frame {frame_index + 1}/{total_frames}: {image_path.name}")

        current_time = args.start_time + frame_index * frame_interval_sec
        image = preprocess_image(
            image_path=image_path,
            roi=roi,
            scale=args.scale,
            contrast=args.contrast,
            sharpness=args.sharpness,
            median_filter_size=args.median_filter_size,
        )

        results = model.track(
            source=image,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=class_ids,
            device=args.device,
            verbose=False,
        )

        summary, next_logical_id = summarize_detections(
            result=results[0],
            current_time=current_time,
            logical_tracks=logical_tracks,
            raw_to_logical=raw_to_logical,
            next_logical_id=next_logical_id,
            frame_index=frame_index,
            max_missing_frames=args.max_missing_frames,
            stitch_iou_threshold=args.stitch_iou_threshold,
            stitch_center_threshold=args.stitch_center_threshold,
        )

        if summary.new_ids:
            last_new_bus_time = current_time

        avg_area = sum(summary.areas) / len(summary.areas) if summary.areas else 0.0
        max_area = max(summary.areas) if summary.areas else 0.0
        total_waiting_time = sum(summary.waiting_times)
        avg_waiting_time = (
            total_waiting_time / len(summary.waiting_times)
            if summary.waiting_times
            else 0.0
        )
        max_waiting_time = max(summary.waiting_times) if summary.waiting_times else 0.0
        seconds_since_last_new_bus = (
            current_time - last_new_bus_time if last_new_bus_time is not None else None
        )

        rows.append(
            {
                "frame_index": frame_index,
                "source_frame": image_path.name,
                "time_second": round(current_time, 3),
                "time_hhmmss": seconds_to_hhmmss(current_time),
                "bus_count_inside": summary.count,
                "bus_count_diff": summary.count - previous_count,
                "bus_ids_inside": ";".join(str(track_id) for track_id in summary.ids),
                "raw_tracker_ids_inside": ";".join(str(track_id) for track_id in summary.raw_ids),
                "new_bus_count": len(summary.new_ids),
                "new_bus_ids": ";".join(str(track_id) for track_id in summary.new_ids),
                "exited_bus_count": len(summary.exited_ids),
                "exited_bus_ids": ";".join(str(track_id) for track_id in summary.exited_ids),
                "recovered_bus_count": len(summary.recovered_ids),
                "recovered_bus_ids": ";".join(str(track_id) for track_id in summary.recovered_ids),
                "raw_id_switch_count": summary.raw_id_switches,
                "avg_area": round(avg_area, 3),
                "avg_area_diff": round(avg_area - previous_avg_area, 3),
                "max_area": round(max_area, 3),
                "remaining_time": round(total_waiting_time, 3),
                "total_waiting_time": round(total_waiting_time, 3),
                "avg_waiting_time": round(avg_waiting_time, 3),
                "max_waiting_time": round(max_waiting_time, 3),
                "seconds_since_last_new_bus": (
                    round(seconds_since_last_new_bus, 3)
                    if seconds_since_last_new_bus is not None
                    else ""
                ),
            }
        )

        previous_count = summary.count
        previous_avg_area = avg_area

    write_csv(rows, output_csv)
    log(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
