from typing import Any, Dict
from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from app_rep import calculate_player_scores
from utils.player_stats_drawer_utils import generate_report_max_only,get_video_fps
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt  
import cv2  
import pandas as pd
from copy import deepcopy
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(input_video_path: str) -> Dict[str, Any]:
    try:
        # Read Video
        FPS = get_video_fps(input_video_path)
        video_frames = read_video(input_video_path)
        
        if not video_frames:
            raise ValueError("No frames could be read from the video")

        # Detect Players and Ball
        player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
        ball_tracker = BallTracker(model_path='models/yolov5su.pt')

        player_detections = player_tracker.detect_frames(video_frames,
                                                         read_from_stub=True,
                                                         stub_path="tracker_stubs/player_detections_original.pkl"
                                                         )
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                         read_from_stub=True,
                                                         stub_path="tracker_stubs/ball_detections_original.pkl"
                                                         )
        
        if not ball_detections or not player_detections:
            raise ValueError("No ball or player detections found")

        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        
        # Court Line Detector model
        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_line_detector.predict(video_frames[0])
        
        if not court_keypoints:
            raise ValueError("Could not detect court keypoints")

        # choose players
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

        # Extract exactly two keys from each dictionary
        filtered_data = [{k: v for i, (k, v) in enumerate(item.items()) if i < 2} for item in player_detections]

        # Convert keys to a list and extract the first two
        keys_list = list(filtered_data[1].keys())
        player_1 = keys_list[0]
        player_2 = keys_list[1]

        # MiniCourt
        mini_court = MiniCourt(video_frames[0]) 

        # Detect ball shots
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
        
        if not ball_shot_frames:
            logger.warning("No ball shots detected in the video")
            ball_shot_frames = [0, len(video_frames)-1]  # Use start and end frames as fallback

        # Convert positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, 
            ball_detections,
            court_keypoints,
            player_1,
            player_2
        )

        # Initialize player stats with safe values
        player_stats_data = [{
            'frame_num': 0,
            f'player_{player_1}_number_of_shots': 0,
            f'player_{player_1}_total_shot_speed': 0,
            f'player_{player_1}_last_shot_speed': 0,
            f'player_{player_1}_total_player_speed': 0,
            f'player_{player_1}_last_player_speed': 0,

            f'player_{player_2}_number_of_shots': 0,
            f'player_{player_2}_total_shot_speed': 0,
            f'player_{player_2}_last_shot_speed': 0,
            f'player_{player_2}_total_player_speed': 0,
            f'player_{player_2}_last_player_speed': 0,
        }]
        
        for ball_shot_ind in range(len(ball_shot_frames)-1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind+1]
            
            if end_frame <= start_frame:
                continue
                
            ball_shot_time_in_seconds = (end_frame-start_frame)/FPS 

            # Get distance covered by the ball
            try:
                distance_covered_by_ball_pixels = measure_distance(
                    ball_mini_court_detections[start_frame][1],
                    ball_mini_court_detections[end_frame][1]
                )
                distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
                    distance_covered_by_ball_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )

                # Speed of the ball shot in km/h
                speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6
                speed_of_ball_shot = min(speed_of_ball_shot, 200)  # Cap at 200 km/h
            except Exception as e:
                logger.warning(f"Error calculating ball speed: {e}")
                speed_of_ball_shot = 0

            # player who shot the ball
            try:
                player_positions = player_mini_court_detections[start_frame]
                player_shot_ball = min(
                    [player_1, player_2], 
                    key=lambda player_id: measure_distance(
                        player_positions[player_id], 
                        ball_mini_court_detections[start_frame][1]
                    )
                )
            except Exception as e:
                logger.warning(f"Error determining shot player: {e}")
                player_shot_ball = player_1  # Default to player 1

            # opponent player speed
            opponent_player_id = player_2 if player_shot_ball == player_1 else player_1

            try:
                distance_covered_by_opponent_pixels = measure_distance(
                    player_mini_court_detections[start_frame][opponent_player_id],
                    player_mini_court_detections[end_frame][opponent_player_id]
                )
                distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                    distance_covered_by_opponent_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )

                speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6
                speed_of_opponent = min(speed_of_opponent, 15)  # Cap at 15 m/s
            except Exception as e:
                logger.warning(f"Error calculating opponent speed: {e}")
                speed_of_opponent = 0

            current_player_stats = deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)

        # Convert to DataFrame and handle missing frames
        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        # Calculate averages with safe division
        for player_id in [player_1, player_2]:
            player_stats_data_df[f'player_{player_id}_average_shot_speed'] = (
                player_stats_data_df[f'player_{player_id}_total_shot_speed'] / 
                player_stats_data_df[f'player_{player_id}_number_of_shots'].replace(0, 1)
            )
            player_stats_data_df[f'player_{player_id}_average_player_speed'] = (
                player_stats_data_df[f'player_{player_id}_total_player_speed'] / 
                player_stats_data_df[f'player_{player_id}_number_of_shots'].replace(0, 1)
            )

        # Draw output
        output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames)
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))    

        # Draw Player Stats
        player_states = draw_player_stats(output_video_frames, player_stats_data_df, player_1, player_2, FPS)
        max_stats_df = generate_report_max_only(player_states, player_1, player_2)
        result = calculate_player_scores(max_stats_df)
        
        return result

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {
            "error": str(e),
            "status": "error"
        }
