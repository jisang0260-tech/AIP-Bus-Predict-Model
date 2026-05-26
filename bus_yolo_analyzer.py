from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_GATE_CONFIG = Path("configs/gate_rois.json")
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


@dataclass(frozen=True)
class GateRoi:
    name: str
    roi: tuple[int, int, int, int]
    out_direction: tuple[float, float]


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


def normalize_direction(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= 0:
        raise ValueError("Gate out direction cannot be zero length")
    return dx / length, dy / length


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def normalize_roi(
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1 = clamp_point(start[0], start[1], width, height)
    x2, y2 = clamp_point(end[0], end[1], width, height)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left < 5 or bottom - top < 5:
        raise ValueError("Gate ROI is too small")
    return left, top, right, bottom


def load_gate_config(config_path: Path | None) -> list[GateRoi]:
    if config_path is None or not config_path.exists():
        return []

    data = json.loads(config_path.read_text(encoding="utf-8"))
    gates_data = data.get("gates", [])
    if not isinstance(gates_data, list):
        raise ValueError(f"Invalid gate config: {config_path}")

    gates: list[GateRoi] = []
    for index, item in enumerate(gates_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid gate entry #{index} in {config_path}")

        roi = item.get("roi")
        direction = item.get("out_direction")
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"Invalid roi for gate #{index} in {config_path}")
        if not isinstance(direction, list) or len(direction) != 2:
            raise ValueError(f"Invalid out_direction for gate #{index} in {config_path}")

        dx, dy = normalize_direction(float(direction[0]), float(direction[1]))
        gates.append(
            GateRoi(
                name=str(item.get("name") or f"gate_{index}"),
                roi=tuple(int(value) for value in roi),
                out_direction=(dx, dy),
            )
        )

    return gates


def draw_gate_overlay(image, gates: list[GateRoi], scale: float = 1.0):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for gate configuration. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    for index, gate in enumerate(gates, start=1):
        x1, y1, x2, y2 = (int(round(value * scale)) for value in gate.roi)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 255), 2)

        cx = int(round((gate.roi[0] + gate.roi[2]) * 0.5 * scale))
        cy = int(round((gate.roi[1] + gate.roi[3]) * 0.5 * scale))
        arrow_len = max(35, int(round(max(x2 - x1, y2 - y1) * 0.35)))
        dx, dy = gate.out_direction
        end = (
            int(round(cx + dx * arrow_len)),
            int(round(cy + dy * arrow_len)),
        )
        cv2.arrowedLine(image, (cx, cy), end, (0, 0, 255), 3, tipLength=0.25)
        cv2.putText(
            image,
            f"{index}: {gate.name} OUT",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def write_gate_config(
    config_path: Path,
    preview_path: Path,
    image_path: Path,
    image_size: tuple[int, int],
    gates: list[GateRoi],
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "coordinate_space": "original_frame",
        "source_frame": str(image_path),
        "preview_frame": str(preview_path),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "gates": [
            {
                "name": gate.name,
                "roi": list(gate.roi),
                "out_direction": [
                    round(gate.out_direction[0], 6),
                    round(gate.out_direction[1], 6),
                ],
            }
            for gate in gates
        ],
    }
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def configure_gate_rois(
    image_path: Path,
    config_path: Path,
    gate_count: int,
    display_width: int,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for gate configuration. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read frame for gate configuration: {image_path}")

    height, width = image.shape[:2]
    scale = min(1.0, display_width / width)
    display_size = (int(round(width * scale)), int(round(height * scale)))
    window_name = "Configure gate ROIs"
    gates: list[GateRoi] = []
    mode = "roi"
    start_point: tuple[int, int] | None = None
    current_point: tuple[int, int] | None = None
    current_roi: tuple[int, int, int, int] | None = None

    def to_original(point: tuple[int, int]) -> tuple[int, int]:
        return clamp_point(
            int(round(point[0] / scale)),
            int(round(point[1] / scale)),
            width,
            height,
        )

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal mode, start_point, current_point, current_roi
        if len(gates) >= gate_count:
            return

        point = to_original((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = point
            current_point = point
            return

        if event == cv2.EVENT_MOUSEMOVE and start_point is not None:
            current_point = point
            return

        if event != cv2.EVENT_LBUTTONUP or start_point is None:
            return

        current_point = point
        if mode == "roi":
            try:
                current_roi = normalize_roi(start_point, current_point, width, height)
            except ValueError as exc:
                log(str(exc))
                current_roi = None
            else:
                mode = "direction"
                log(
                    f"Gate {len(gates) + 1}: drag an arrow in the OUT direction."
                )
        else:
            dx = current_point[0] - start_point[0]
            dy = current_point[1] - start_point[1]
            if math.hypot(dx, dy) < 10:
                log("Direction arrow is too short. Drag a longer OUT direction arrow.")
            elif current_roi is not None:
                out_direction = normalize_direction(dx, dy)
                gates.append(
                    GateRoi(
                        name=f"gate_{len(gates) + 1}",
                        roi=current_roi,
                        out_direction=out_direction,
                    )
                )
                log(f"Saved gate {len(gates)}/{gate_count}.")
                current_roi = None
                mode = "roi"

        start_point = None
        current_point = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])
    cv2.setMouseCallback(window_name, on_mouse)
    log("Draw each gate ROI, then drag the arrow in the OUT direction.")
    log("Keys: u=undo, r=reset current, q/esc=cancel, enter=save when complete.")
    completion_logged = False

    while True:
        canvas = image.copy()
        draw_gate_overlay(canvas, gates)

        if current_roi is not None:
            x1, y1, x2, y2 = current_roi
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)

        if start_point is not None and current_point is not None:
            if mode == "roi":
                try:
                    x1, y1, x2, y2 = normalize_roi(start_point, current_point, width, height)
                except ValueError:
                    x1, y1 = start_point
                    x2, y2 = current_point
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
            else:
                cv2.arrowedLine(
                    canvas,
                    start_point,
                    current_point,
                    (0, 0, 255),
                    3,
                    tipLength=0.25,
                )

        if len(gates) >= gate_count:
            instruction = "All gates set. Press Enter to save."
        else:
            instruction = (
                f"Gate {len(gates) + 1}/{gate_count}: "
                + ("draw ROI rectangle" if mode == "roi" else "drag OUT arrow")
            )
        cv2.putText(
            canvas,
            instruction,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "u undo | r reset current | enter save | q cancel",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display = cv2.resize(canvas, display_size, interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key in {ord("q"), 27}:
            cv2.destroyWindow(window_name)
            raise SystemExit("Gate configuration cancelled.")
        if key == ord("u") and gates:
            removed = gates.pop()
            completion_logged = False
            log(f"Removed {removed.name}.")
        if key == ord("r"):
            mode = "roi"
            start_point = None
            current_point = None
            current_roi = None
            log("Reset current gate drawing.")
        if key in {13, 10} and len(gates) >= gate_count:
            break
        if len(gates) >= gate_count and not completion_logged:
            log("All gates are set. Press Enter to save, or u to undo.")
            completion_logged = True

    preview_path = config_path.with_name(config_path.stem + "_preview.jpg")
    preview = draw_gate_overlay(image.copy(), gates)
    cv2.imwrite(str(preview_path), preview)
    cv2.destroyWindow(window_name)
    write_gate_config(
        config_path=config_path,
        preview_path=preview_path,
        image_path=image_path,
        image_size=(width, height),
        gates=gates,
    )
    log(f"Saved gate config to {config_path}")
    log(f"Saved gate preview to {preview_path}")


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
    parser.add_argument(
        "--gate-config",
        default=DEFAULT_GATE_CONFIG,
        type=Path,
        help="JSON file containing gate ROI boxes and OUT directions.",
    )
    parser.add_argument(
        "--configure-gates",
        action="store_true",
        help="Open a video frame and interactively draw gate ROIs and OUT directions.",
    )
    parser.add_argument(
        "--gate-count",
        default=2,
        type=int,
        help="Number of gate ROIs to configure. Default is 2.",
    )
    parser.add_argument(
        "--gate-frame-index",
        default=0,
        type=int,
        help="Frame index to display when configuring gates.",
    )
    parser.add_argument(
        "--gate-display-width",
        default=1280,
        type=int,
        help="Maximum display width for the interactive gate configuration window.",
    )
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

    if args.gate_count <= 0:
        raise ValueError("--gate-count must be greater than zero")

    if args.gate_frame_index < 0:
        raise ValueError("--gate-frame-index cannot be negative")

    if args.gate_display_width <= 0:
        raise ValueError("--gate-display-width must be greater than zero")

    if not 0 <= args.stitch_iou_threshold <= 1:
        raise ValueError("--stitch-iou-threshold must be between 0 and 1")

    if args.stitch_center_threshold < 0:
        raise ValueError("--stitch-center-threshold cannot be negative")

    roi = None if args.no_roi or args.roi is None else parse_int_tuple(args.roi, 4, "--roi")
    class_ids = parse_classes(args.classes)
    image_paths, output_csv, frame_interval_sec = resolve_source(args)

    if args.configure_gates:
        if not image_paths:
            raise FileNotFoundError(
                f"No images found with pattern {args.pattern}. Check the video or frame folder."
            )
        if args.gate_frame_index >= len(image_paths):
            raise ValueError(
                f"--gate-frame-index {args.gate_frame_index} is outside "
                f"the available frame range 0-{len(image_paths) - 1}"
            )
        configure_gate_rois(
            image_path=image_paths[args.gate_frame_index],
            config_path=args.gate_config,
            gate_count=args.gate_count,
            display_width=args.gate_display_width,
        )
        return

    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found with pattern {args.pattern}. Check the video or frame folder."
        )

    gate_rois = load_gate_config(args.gate_config)
    if gate_rois:
        gate_names = ", ".join(gate.name for gate in gate_rois)
        log(f"Loaded {len(gate_rois)} gate ROI(s) from {args.gate_config}: {gate_names}")

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
