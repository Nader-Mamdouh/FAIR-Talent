import math
import sys
from typing import Dict, List, Tuple

import cv2
import pickle
from ultralytics import YOLO

sys.path.append("../")
from utils import measure_distance, get_center_of_bbox  # type: ignore

# Bounding-box filter thresholds
MIN_PLAYER_AREA = 0          # minimum bbox area for a valid player
MAX_PLAYER_AREA = 50000      # maximum bbox area for a valid player
MIN_ASPECT_RATIO = 0.2       # min height/width ratio
MAX_ASPECT_RATIO = 10.0      # max height/width ratio


class PlayerTracker:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def choose_and_filter_players(
        self,
        court_keypoints: List[float],
        player_detections: List[Dict[int, List[float]]],
    ) -> List[Dict[int, List[float]]]:
        """
        Given detections for all frames, choose two players from the first frame
        and filter all frames down to those two track IDs.
        """
        if not player_detections:
            raise ValueError("No player detections available.")

        first_frame_detections = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, first_frame_detections)

        filtered_player_detections: List[Dict[int, List[float]]] = []
        for frame_dict in player_detections:
            filtered_dict = {
                track_id: bbox
                for track_id, bbox in frame_dict.items()
                if track_id in chosen_players
            }
            filtered_player_detections.append(filtered_dict)

        return filtered_player_detections

    def choose_players(
        self,
        court_keypoints: List[float],
        player_dict: Dict[int, List[float]],
    ) -> List[int]:
        """
        Choose two players based on:
        - bbox size and aspect ratio
        - distance to court keypoints (closer is better)
        Returns raw tracker IDs.
        """
        if not player_dict:
            raise ValueError("Player dictionary is empty.")

        player_scores: List[Tuple[int, float]] = []

        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            aspect_ratio = bbox_height / bbox_width if bbox_width != 0 else math.inf

            # Filter by size/aspect ratio
            if (
                bbox_area < MIN_PLAYER_AREA
                or bbox_area > MAX_PLAYER_AREA
                or aspect_ratio < MIN_ASPECT_RATIO
                or aspect_ratio > MAX_ASPECT_RATIO
            ):
                continue

            # Distance to nearest court keypoint
            player_center = get_center_of_bbox(bbox)
            min_distance = math.inf
            for i in range(0, len(court_keypoints), 2):
                court_pt = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_pt)
                min_distance = min(min_distance, distance)

            # Lower score is better: closer to court & larger bbox
            score = min_distance / bbox_area if bbox_area != 0 else math.inf
            player_scores.append((track_id, score))

        if len(player_scores) < 2:
            raise ValueError("Not enough players to choose from.")

        player_scores.sort(key=lambda x: x[1])
        chosen_players = [player_scores[0][0], player_scores[1][0]]
        return chosen_players

    def detect_frames(
        self,
        frames: List,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> List[Dict[int, List[float]]]:
        """
        Run detection over a list of frames.
        If read_from_stub is True and stub_path is given, load detections from pickle instead.
        """
        player_detections: List[Dict[int, List[float]]] = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame) -> Dict[int, List[float]]:
        """
        Detect players in a single frame using YOLO tracking.
        Returns a dict: {track_id: [x1, y1, x2, y2]} for 'person' class only.
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict: Dict[int, List[float]] = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            if cls_name == "person":
                player_dict[track_id] = bbox

        return player_dict

    def draw_bboxes(
        self,
        video_frames: List,
        player_detections: List[Dict[int, List[float]]],
    ) -> List:
        """
        Draw bounding boxes and raw track IDs on frames.
        """
        output_video_frames: List = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
            output_video_frames.append(frame)

        return output_video_frames