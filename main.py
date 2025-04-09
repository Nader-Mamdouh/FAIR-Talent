import cv2
import pandas as pd
from copy import deepcopy
import numpy as np
from typing import Dict, Any

from utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters
)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants

def process_video(input_video_path: str) -> Dict[str, Any]:
    """Main video processing pipeline."""
    try:
        # Initialize trackers
        player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
        ball_tracker = BallTracker(model_path='models/yolov5s.pt')
        
        # Read and process video
        video_frames = read_video(input_video_path)
        player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections_original.pkl"
                                                     )
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                        read_from_stub=False,
                                                        stub_path="tracker_stubs/ball_detections_original.pkl"
                                                        )
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

        # Court detection
        court_model = CourtLineDetector("models/keypoints_model.pth")
        court_keypoints = court_model.predict(video_frames[0])
        
        # Player selection
        player_detections = player_tracker.choose_and_filter_players(
            court_keypoints, player_detections
        )

        # Get player IDs
        player_ids = list(player_detections[0].keys())
        if len(player_ids) < 2:
            return {
                "error": "Could not detect two players",
                "detected_players": player_ids
            }
            
        player_1, player_2 = player_ids[:2]

        # MiniCourt processing
        mini_court = MiniCourt(video_frames[0])
        player_mini_court, ball_mini_court = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints, player_1, player_2
        )

        # Shot detection
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

        # Statistics calculation
        stats = calculate_match_statistics(
            ball_shot_frames,
            player_mini_court,
            ball_mini_court,
            player_1,
            player_2,
            mini_court
        )

        return {
            "status": "success",
            "players": {player_1: "Player 1", player_2: "Player 2"},
            "statistics": stats,
            "total_shots": len(ball_shot_frames) - 1
        }

    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }

def calculate_match_statistics(ball_shot_frames, player_pos, ball_pos, p1, p2, court):
    """Calculate player statistics from tracking data."""
    stats = {
        p1: initialize_player_stats(),
        p2: initialize_player_stats()
    }

    for i in range(len(ball_shot_frames)-1):
        start, end = ball_shot_frames[i], ball_shot_frames[i+1]
        duration = (end - start) / 24  # Assuming 24fps

        # Determine who hit the ball
        hitter = determine_hitter(player_pos[start], ball_pos[start], p1, p2)
        defender = p2 if hitter == p1 else p1

        # Calculate ball speed
        ball_dist = measure_distance(ball_pos[start][1], ball_pos[end][1])
        ball_dist_m = convert_pixel_distance_to_meters(
            ball_dist, constants.DOUBLE_LINE_WIDTH, court.get_width_of_mini_court()
        )
        ball_speed = (ball_dist_m / duration) * 3.6  # km/h

        # Calculate player movement
        player_dist = measure_distance(
            player_pos[start][defender], player_pos[end][defender]
        )
        player_dist_m = convert_pixel_distance_to_meters(
            player_dist, constants.DOUBLE_LINE_WIDTH, court.get_width_of_mini_court()
        )
        player_speed = (player_dist_m / duration) * 3.6  # km/h

        # Update stats
        stats[hitter]["shots"] += 1
        stats[hitter]["total_shot_speed"] += ball_speed
        stats[hitter]["last_shot_speed"] = ball_speed
        
        stats[defender]["total_movement"] += player_dist_m
        stats[defender]["last_speed"] = player_speed

    # Calculate averages
    for player in [p1, p2]:
        if stats[player]["shots"] > 0:
            stats[player]["avg_shot_speed"] = float(
                stats[player]["total_shot_speed"] / stats[player]["shots"]
            )
        if stats[player]["total_movement"] > 0:
            stats[player]["avg_speed"] = float(
                stats[player]["total_movement"] / (len(ball_shot_frames)-1)
            )
        # Convert all numpy values to Python floats
        stats[player]["total_shot_speed"] = float(stats[player]["total_shot_speed"])
        stats[player]["last_shot_speed"] = float(stats[player]["last_shot_speed"])
        stats[player]["total_movement"] = float(stats[player]["total_movement"])
        stats[player]["last_speed"] = float(stats[player]["last_speed"])

    # Convert stats to DataFrame format
    player_stats_data_df = pd.DataFrame()
    for frame_idx in range(len(ball_shot_frames)-1):
        start = ball_shot_frames[frame_idx]
        end = ball_shot_frames[frame_idx+1]
        hitter = determine_hitter(player_pos[start], ball_pos[start], p1, p2)
        defender = p2 if hitter == p1 else p1
        
        # Calculate speeds
        ball_dist = measure_distance(ball_pos[start][1], ball_pos[end][1])
        ball_dist_m = convert_pixel_distance_to_meters(
            ball_dist, constants.DOUBLE_LINE_WIDTH, court.get_width_of_mini_court()
        )
        ball_speed = (ball_dist_m / (end - start) / 24) * 3.6  # km/h
        
        player_dist = measure_distance(
            player_pos[start][defender], player_pos[end][defender]
        )
        player_dist_m = convert_pixel_distance_to_meters(
            player_dist, constants.DOUBLE_LINE_WIDTH, court.get_width_of_mini_court()
        )
        player_speed = (player_dist_m / (end - start) / 24) * 3.6  # km/h
        
        # Update DataFrame
        current_stats = {
            f'player_{hitter}_number_of_shots': 1,
            f'player_{hitter}_total_shot_speed': ball_speed,
            f'player_{hitter}_last_shot_speed': ball_speed,
            f'player_{defender}_total_player_speed': player_speed,
            f'player_{defender}_last_player_speed': player_speed
        }
        player_stats_data_df = pd.concat([player_stats_data_df, pd.DataFrame([current_stats])], ignore_index=True)
    
    # Calculate averages
    for player in [p1, p2]:
        player_stats_data_df[f'player_{player}_average_shot_speed'] = (
            player_stats_data_df[f'player_{player}_total_shot_speed'] / 
            player_stats_data_df[f'player_{player}_number_of_shots'].replace(0, 1)
        )
        player_stats_data_df[f'player_{player}_average_player_speed'] = (
            player_stats_data_df[f'player_{player}_total_player_speed'] / 
            player_stats_data_df[f'player_{player}_number_of_shots'].replace(0, 1)
        )

    return stats

def initialize_player_stats():
    return {
        "shots": 0,
        "total_shot_speed": 0,
        "last_shot_speed": 0,
        "avg_shot_speed": 0,
        "total_movement": 0,
        "last_speed": 0,
        "avg_speed": 0
    }

def determine_hitter(player_pos, ball_pos, p1, p2):
    """Determine which player hit the ball."""
    dist_p1 = measure_distance(player_pos[p1], ball_pos[1])
    dist_p2 = measure_distance(player_pos[p2], ball_pos[1])
    return p1 if dist_p1 < dist_p2 else p2