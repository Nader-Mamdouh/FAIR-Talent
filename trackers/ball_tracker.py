from typing import List, Dict, Tuple

import cv2
import pickle
import pandas as pd
from ultralytics import YOLO


class BallTracker:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        # Slightly higher confidence to reduce false positives
        self.confidence_threshold = 0.15

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, List[float]]]) -> List[Dict[int, List[float]]]:
        """
        Fill missing ball positions across frames by interpolating bounding boxes.
        Assumes the ball is stored under key 1 in each detection dict.
        """
        # Extract ball bbox per frame (or [] if missing)
        raw_positions = [frame_dict.get(1, []) for frame_dict in ball_positions]

        # Build DataFrame with columns x1, y1, x2, y2
        df_ball = pd.DataFrame(raw_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values, then back/forward fill any remaining gaps
        df_ball = df_ball.interpolate()
        df_ball = df_ball.bfill()
        df_ball = df_ball.ffill()

        # Convert back to list[dict] with ball ID 1
        interpolated = [{1: bbox} for bbox in df_ball.to_numpy().tolist()]
        return interpolated

    def get_ball_shot_frames(self, ball_positions: List[Dict[int, List[float]]]) -> List[int]:
        """
        Detect frames where the ball is hit, based on changes in vertical position.
        Returns list of frame indices with 'ball_hit' == 1.
        """
        raw_positions = [frame_dict.get(1, []) for frame_dict in ball_positions]
        df_ball = pd.DataFrame(raw_positions, columns=["x1", "y1", "x2", "y2"])

        df_ball["ball_hit"] = 0

        # Midpoint of bbox in Y direction
        df_ball["mid_y"] = (df_ball["y1"] + df_ball["y2"]) / 2.0
        df_ball["mid_y_rolling_mean"] = df_ball["mid_y"].rolling(
            window=5, min_periods=1, center=False
        ).mean()
        df_ball["delta_y"] = df_ball["mid_y_rolling_mean"].diff()

        # Shot detection parameters (lenient)
        minimum_change_frames_for_hit = 8
        min_delta_y_threshold = 2.0

        max_window = int(minimum_change_frames_for_hit * 1.2)
        for i in range(1, len(df_ball) - max_window):
            dy_i = df_ball["delta_y"].iloc[i]
            dy_next = df_ball["delta_y"].iloc[i + 1]

            negative_change = dy_i > min_delta_y_threshold and dy_next < -min_delta_y_threshold
            positive_change = dy_i < -min_delta_y_threshold and dy_next > min_delta_y_threshold

            if not (negative_change or positive_change):
                continue

            change_count = 0
            for change_frame in range(i + 1, i + max_window + 1):
                if change_frame >= len(df_ball):
                    break

                dy_follow = df_ball["delta_y"].iloc[change_frame]

                neg_follow = dy_i > min_delta_y_threshold and dy_follow < -min_delta_y_threshold
                pos_follow = dy_i < -min_delta_y_threshold and dy_follow > min_delta_y_threshold

                if negative_change and neg_follow:
                    change_count += 1
                elif positive_change and pos_follow:
                    change_count += 1

            if change_count > minimum_change_frames_for_hit - 1:
                df_ball.loc[i, "ball_hit"] = 1

        hit_frames = df_ball[df_ball["ball_hit"] == 1].index.tolist()
        return hit_frames

    def detect_frames(
        self,
        frames: List,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> List[Dict[int, List[float]]]:
        """
        Run ball detection across frames, optionally loading/saving from a pickle stub.
        """
        ball_detections: List[Dict[int, List[float]]] = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame) -> Dict[int, List[float]]:
        """
        Detect ball in a single frame. Returns {1: [x1, y1, x2, y2]} if a box exists,
        otherwise {} (no ball).
        """
        results = self.model.predict(frame, conf=self.confidence_threshold)[0]

        ball_dict: Dict[int, List[float]] = {}
        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            # For now, we treat the first detected box as the ball and assign ID 1
            ball_dict[1] = bbox

        return ball_dict

    def draw_bboxes(
        self,
        video_frames: List,
        ball_detections: List[Dict[int, List[float]]],
    ) -> List:
        """
        Draw ball bounding boxes on frames.
        """
        output_video_frames: List = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2,
                )
            output_video_frames.append(frame)

        return output_video_frames