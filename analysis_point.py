import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import json

def generate_json_stats(player_1, player_2, player_stats_df):
    """Generate statistics in JSON format"""
    
    # Calculate statistics for each player
    stats = {
        "player_1": {
            "id": player_1,
            "number_of_shots": int(player_stats_df[f'player_{player_1}_number_of_shots'].max()),
            "average_shot_speed": float(player_stats_df[f'player_{player_1}_average_shot_speed'].mean()),
            "max_shot_speed": float(player_stats_df[f'player_{player_1}_last_shot_speed'].max()),
            "average_player_speed": float(player_stats_df[f'player_{player_1}_average_player_speed'].mean()),
            "max_player_speed": float(player_stats_df[f'player_{player_1}_last_player_speed'].max())
        },
        "player_2": {
            "id": player_2,
            "number_of_shots": int(player_stats_df[f'player_{player_2}_number_of_shots'].max()),
            "average_shot_speed": float(player_stats_df[f'player_{player_2}_average_shot_speed'].mean()),
            "max_shot_speed": float(player_stats_df[f'player_{player_2}_last_shot_speed'].max()),
            "average_player_speed": float(player_stats_df[f'player_{player_2}_average_player_speed'].mean()),
            "max_player_speed": float(player_stats_df[f'player_{player_2}_last_player_speed'].max())
        },
        "match_summary": {
            "total_frames": len(player_stats_df),
            "match_duration_seconds": len(player_stats_df) / 24,  # assuming 24fps
            "total_shots": int(player_stats_df[f'player_{player_1}_number_of_shots'].max() + 
                             player_stats_df[f'player_{player_2}_number_of_shots'].max())
        }
    }
    
    return stats
