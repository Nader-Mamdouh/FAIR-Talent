import os
from typing import Any, Dict

from utils import (
    read_video,
    save_video,  # if you don't use this, you can remove the import
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
)
import constants
from app_rep import calculate_player_scores
from utils.player_stats_drawer_utils import generate_report_max_only, get_video_fps
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

import cv2
import pandas as pd
from copy import deepcopy
import numpy as np
import _osx_support

def process_video(input_video_path: str) -> Dict[str, Any]:
    # Read video and FPS
    FPS = get_video_fps(input_video_path)
    video_frames = read_video(input_video_path)

    if not video_frames:
        raise ValueError("No frames read from video")

    # Detect players and ball (using stubs for now)
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/best.pt")

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections_original.pkl",
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections_original.pkl",
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court line detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players (raw tracker IDs)
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # Extract two player IDs from detections
    filtered_data = [
        {k: v for i, (k, v) in enumerate(item.items()) if i < 2}
        for item in player_detections
    ]
    if len(filtered_data) < 2:
        raise ValueError("Not enough frames with player detections to choose two players")

    keys_list = list(filtered_data[1].keys())
    if len(keys_list) < 2:
        raise ValueError("Frame 1 does not contain two distinct players")

    raw_player_1 = keys_list[0]
    raw_player_2 = keys_list[1]
    print("raw player IDs:", raw_player_1, raw_player_2)

    # Canonical logical IDs: player_1 and player_2
    canonical_p1 = 1
    canonical_p2 = 2
    id_map = {
        raw_player_1: canonical_p1,
        raw_player_2: canonical_p2,
    }

    # Mini court
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court coordinates (still using raw IDs)
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections,
            ball_detections,
            court_keypoints,
            raw_player_1,
            raw_player_2,
        )
    )

    # Initialize stats with canonical player_1 / player_2 columns
    player_stats_data = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0.0,
            "player_1_last_shot_speed": 0.0,
            "player_1_total_player_speed": 0.0,
            "player_1_last_player_speed": 0.0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0.0,
            "player_2_last_shot_speed": 0.0,
            "player_2_total_player_speed": 0.0,
            "player_2_last_player_speed": 0.0,
        }
    ]

    # Aggregate stats across ball shots
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        if end_frame <= start_frame:
            continue

        ball_shot_time_in_seconds = (end_frame - start_frame) / FPS

        # Ball distance and speed
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )
        speed_of_ball_shot = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )

        # Player who shot the ball (raw ID)
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            [raw_player_1, raw_player_2],
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )
        opponent_raw_id = raw_player_2 if player_shot_ball == raw_player_1 else raw_player_1

        # Canonical IDs for stats
        shooter_canonical = id_map[player_shot_ball]
        opponent_canonical = canonical_p2 if shooter_canonical == canonical_p1 else canonical_p1

        shooter_prefix = f"player_{shooter_canonical}"
        opponent_prefix = f"player_{opponent_canonical}"

        # Opponent movement distance and speed
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_raw_id],
            player_mini_court_detections[end_frame][opponent_raw_id],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )
        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        # Update stats
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame

        current_player_stats[f"{shooter_prefix}_number_of_shots"] += 1
        current_player_stats[f"{shooter_prefix}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"{shooter_prefix}_last_shot_speed"] = speed_of_ball_shot

        current_player_stats[f"{opponent_prefix}_total_player_speed"] += speed_of_opponent
        current_player_stats[f"{opponent_prefix}_last_player_speed"] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # Build full stats DataFrame (one row per frame)
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on="frame_num", how="left"
    )
    player_stats_data_df = player_stats_data_df.ffill()

    # Average shot speed and player speed for both players (canonical 1 and 2)
    for pid, other_pid in [(canonical_p1, canonical_p2), (canonical_p2, canonical_p1)]:
        prefix = f"player_{pid}"
        other_prefix = f"player_{other_pid}"

        shots_col = f"{prefix}_number_of_shots"
        total_shot_col = f"{prefix}_total_shot_speed"
        total_speed_col = f"{prefix}_total_player_speed"
        other_shots_col = f"{other_prefix}_number_of_shots"

        player_stats_data_df[f"{prefix}_average_shot_speed"] = (
            player_stats_data_df[total_shot_col]
            / player_stats_data_df[shots_col].replace(0, 1)
        )
        player_stats_data_df[f"{prefix}_average_player_speed"] = (
            player_stats_data_df[total_speed_col]
            / player_stats_data_df[other_shots_col].replace(0, 1)
        )

    # Draw output overlays
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames)

    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
    )

    # ******** IMPORTANT: add this call ********
    # This mutates player_stats_data_df, adding acceleration, inconsistency, distance, etc.
    draw_player_stats(output_video_frames, player_stats_data_df, canonical_p1, canonical_p2, FPS)

    # Save annotated video locally
    os.makedirs("output", exist_ok=True)
    annotated_video_path = os.path.join("output", "annotated_tennis_2.avi")
    save_video(output_video_frames, annotated_video_path, FPS)

    max_stats_df = generate_report_max_only(player_stats_data_df, canonical_p1, canonical_p2)
    result = calculate_player_scores(max_stats_df)
    result["annotated_video_path"] = annotated_video_path
    return result