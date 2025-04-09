import numpy as np
import cv2
import csv
def draw_player_stats(output_video_frames, player_stats, player_1, player_2):
    for index, row in player_stats.iterrows():
        if index >= len(output_video_frames):
            continue  

        # Handle NaN values to prevent formatting errors
        def safe_value(val):
            return val if not np.isnan(val) else 0  

        # Store extracted values in a dictionary
        player_stats_dict = {
            "shot_speed_1": safe_value(row[f'player_{player_1}_last_shot_speed']),
            "shot_speed_2": safe_value(row[f'player_{player_2}_last_shot_speed']),
            "speed_1": safe_value(row[f'player_{player_1}_last_player_speed']),
            "speed_2": safe_value(row[f'player_{player_2}_last_player_speed']),
            "avg_shot_speed_1": safe_value(row[f'player_{player_1}_average_shot_speed']),
            "avg_shot_speed_2": safe_value(row[f'player_{player_2}_average_shot_speed']),
            "avg_speed_1": safe_value(row[f'player_{player_1}_average_player_speed']),
            "avg_speed_2": safe_value(row[f'player_{player_2}_average_player_speed']),
        }

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        # Rectangle dimensions
        width = 250
        height = 130
        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Header
        output_video_frames[index] = cv2.putText(frame, f"     Player {player_1}     Player {player_2}", (start_x+80, start_y+30), 
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Stats
        stats = [
            ("Shot Speed", player_stats_dict["shot_speed_1"], player_stats_dict["shot_speed_2"]),
            ("Player Speed", player_stats_dict["speed_1"], player_stats_dict["speed_2"]),
            ("Avg. S. Speed", player_stats_dict["avg_shot_speed_1"], player_stats_dict["avg_shot_speed_2"]),
            ("Avg. P. Speed", player_stats_dict["avg_speed_1"], player_stats_dict["avg_speed_2"])
        ]

        for i, (label, p1_value, p2_value) in enumerate(stats):
            y_offset = start_y + 80 + (i * 40)  # Adjust vertical spacing
            output_video_frames[index] = cv2.putText(frame, label, (start_x+10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{p1_value:.1f} km/h    {p2_value:.1f} km/h"
            output_video_frames[index] = cv2.putText(frame, text, (start_x+130, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    output_file="game_report.csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", f"Player {player_1} Shot Speed (km/h)", f"Player {player_2} Shot Speed (km/h)",
                         f"Player {player_1} Player Speed (km/h)", f"Player {player_2} Player Speed (km/h)",
                         f"Player {player_1} Avg Shot Speed (km/h)", f"Player {player_2} Avg Shot Speed (km/h)",
                         f"Player {player_1} Avg Player Speed (km/h)", f"Player {player_2} Avg Player Speed (km/h)"])
        
        for index, row in player_stats.iterrows():
            writer.writerow([
                index,
                row.get(f'player_{player_1}_last_shot_speed', 0),
                row.get(f'player_{player_2}_last_shot_speed', 0),
                row.get(f'player_{player_1}_last_player_speed', 0),
                row.get(f'player_{player_2}_last_player_speed', 0),
                row.get(f'player_{player_1}_average_shot_speed', 0),
                row.get(f'player_{player_2}_average_shot_speed', 0),
                row.get(f'player_{player_1}_average_player_speed', 0),
                row.get(f'player_{player_2}_average_player_speed', 0),
            ])
    print(f"Game report saved to {output_file}")
    
    return output_video_frames

def generate_report(player_stats, player_1, player_2):
    output_file="game_report.csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", f"Player {player_1} Shot Speed (km/h)", f"Player {player_2} Shot Speed (km/h)",
                         f"Player {player_1} Player Speed (km/h)", f"Player {player_2} Player Speed (km/h)",
                         f"Player {player_1} Avg Shot Speed (km/h)", f"Player {player_2} Avg Shot Speed (km/h)",
                         f"Player {player_1} Avg Player Speed (km/h)", f"Player {player_2} Avg Player Speed (km/h)"])
        
        for index, row in player_stats.iterrows():
            writer.writerow([
                index,
                row.get(f'player_{player_1}_last_shot_speed', 0),
                row.get(f'player_{player_2}_last_shot_speed', 0),
                row.get(f'player_{player_1}_last_player_speed', 0),
                row.get(f'player_{player_2}_last_player_speed', 0),
                row.get(f'player_{player_1}_average_shot_speed', 0),
                row.get(f'player_{player_2}_average_shot_speed', 0),
                row.get(f'player_{player_1}_average_player_speed', 0),
                row.get(f'player_{player_2}_average_player_speed', 0),
            ])
    print(f"Game report saved to {output_file}")
    return output_file