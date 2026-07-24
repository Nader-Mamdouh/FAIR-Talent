import numpy as np
import cv2
import pandas as pd


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def safe_value(value: float) -> float:
    """Convert NaN/inf/missing to a safe float value (0.0)."""
    if pd.isna(value) or isinstance(value, float) and np.isnan(value):
        return 0.0
    return float(value)


def calculate_average_speed(player_stats: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Compute average player movement speed and shot speed
    for canonical player_1 and player_2 columns.
    """

    # Player 1 averages
    s1_speed = player_stats["player_1_last_player_speed"].replace(0, np.nan)
    s2_speed = player_stats["player_2_last_player_speed"].replace(0, np.nan)

    s1_shot = player_stats["player_1_last_shot_speed"].replace(0, np.nan)
    s2_shot = player_stats["player_2_last_shot_speed"].replace(0, np.nan)

    player_1_avg_speed = safe_value(s1_speed.mean())
    player_2_avg_speed = safe_value(s2_speed.mean())

    player_1_avg_shot_speed = safe_value(s1_shot.mean())
    player_2_avg_shot_speed = safe_value(s2_shot.mean())

    return player_1_avg_speed, player_2_avg_speed, player_1_avg_shot_speed, player_2_avg_shot_speed


def draw_player_stats(
    output_video_frames: list,
    player_stats: pd.DataFrame,
    player_1: int,  # canonical, expected 1
    player_2: int,  # canonical, expected 2
    FPS: float,
) -> None:
    """
    Enrich player_stats DataFrame with derived metrics for player_1 and player_2.
    Assumes canonical columns: player_1_*, player_2_*.
    """

    # Acceleration (frame-to-frame speed change)
    player_stats["player_1_acceleration"] = (
        player_stats["player_1_last_player_speed"].diff().fillna(0.0)
    )
    player_stats["player_2_acceleration"] = (
        player_stats["player_2_last_player_speed"].diff().fillna(0.0)
    )

    # Shot inconsistency (rolling std of shot speed)
    player_stats["player_1_shot_inconsistency"] = (
        player_stats["player_1_last_shot_speed"]
        .dropna()
        .rolling(5, min_periods=1)
        .std()
        .fillna(0.0)
    )
    player_stats["player_2_shot_inconsistency"] = (
        player_stats["player_2_last_shot_speed"]
        .dropna()
        .rolling(5, min_periods=1)
        .std()
        .fillna(0.0)
    )

    # Distance covered (speed * time, cumulative)
    frame_time = 1.0 / FPS

    player_stats["player_1_distance_covered"] = (
        player_stats["player_1_last_player_speed"] * frame_time
    ).cumsum()
    player_stats["player_2_distance_covered"] = (
        player_stats["player_2_last_player_speed"] * frame_time
    ).cumsum()

    # Rally contribution (count of positive shot-speed deltas)
    player_stats["player_1_rally_contribution"] = (
        player_stats["player_1_last_shot_speed"].diff() > 0
    ).astype(int).cumsum()
    player_stats["player_2_rally_contribution"] = (
        player_stats["player_2_last_shot_speed"].diff() > 0
    ).astype(int).cumsum()

    # Total shots (count of positive shot-speed deltas)
    player_stats["player_1_total_shots"] = (
        player_stats["player_1_last_shot_speed"]
        .diff()
        .gt(0)
        .cumsum()
    )
    player_stats["player_2_total_shots"] = (
        player_stats["player_2_last_shot_speed"]
        .diff()
        .gt(0)
        .cumsum()
    )

    # Rally percentage (contribution / total shots)
    player_stats["player_1_rally_percentage"] = (
        (player_stats["player_1_rally_contribution"] /
         player_stats["player_1_total_shots"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0) * 100.0
    )
    player_stats["player_2_rally_percentage"] = (
        (player_stats["player_2_rally_contribution"] /
         player_stats["player_2_total_shots"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0) * 100.0
    )

    # If you want to use player_stats_dict for drawing overlays, do it here.
    # This loop currently only builds the dict; it doesn't modify frames.
    for index, row in player_stats.iterrows():
        if index >= len(output_video_frames):
            continue

        player_stats_dict = {
            "shot_speed_1": safe_value(row["player_1_last_shot_speed"]),
            "shot_speed_2": safe_value(row["player_2_last_shot_speed"]),
            "speed_1": safe_value(row["player_1_last_player_speed"]),
            "speed_2": safe_value(row["player_2_last_player_speed"]),
            "acceleration_1": safe_value(row["player_1_acceleration"]),
            "acceleration_2": safe_value(row["player_2_acceleration"]),
            "shot_inconsistency_1": safe_value(row["player_1_shot_inconsistency"]),
            "shot_inconsistency_2": safe_value(row["player_2_shot_inconsistency"]),
            "distance_covered_1": safe_value(row["player_1_distance_covered"]),
            "distance_covered_2": safe_value(row["player_2_distance_covered"]),
            "rally_contribution_1": safe_value(row["player_1_rally_contribution"]),
            "rally_contribution_2": safe_value(row["player_2_rally_contribution"]),
        }

        # TODO: use player_stats_dict to draw overlays on output_video_frames[index]
        # e.g., cv2.putText(...) etc.


def generate_report_max_only(player_stats: pd.DataFrame, player_1: int, player_2: int) -> pd.DataFrame:
    """
    Generate a one-row DataFrame with max/avg stats for player_1 and player_2.
    Assumes canonical columns player_1_* and player_2_* are present.
    """

    avg_speed_1, avg_speed_2, avg_shot_speed_1, avg_shot_speed_2 = calculate_average_speed(player_stats)

    max_stats = {
        f"player_{player_1}_max_shot_speed": [safe_value(player_stats["player_1_last_shot_speed"].max())],
        f"player_{player_2}_max_shot_speed": [safe_value(player_stats["player_2_last_shot_speed"].max())],
        f"player_{player_1}_avg_shot_speed": [avg_shot_speed_1],
        f"player_{player_2}_avg_shot_speed": [avg_shot_speed_2],
        f"player_{player_1}_max_speed": [safe_value(player_stats["player_1_last_player_speed"].max())],
        f"player_{player_2}_max_speed": [safe_value(player_stats["player_2_last_player_speed"].max())],
        f"player_{player_1}_avg_speed": [avg_speed_1],
        f"player_{player_2}_avg_speed": [avg_speed_2],
        f"player_{player_1}_max_acceleration": [safe_value(player_stats["player_1_acceleration"].max())],
        f"player_{player_2}_max_acceleration": [safe_value(player_stats["player_2_acceleration"].max())],
        f"player_{player_1}_max_shot_inconsistency": [safe_value(player_stats["player_1_shot_inconsistency"].max())],
        f"player_{player_2}_max_shot_inconsistency": [safe_value(player_stats["player_2_shot_inconsistency"].max())],
        f"player_{player_1}_max_distance_covered": [safe_value(player_stats["player_1_distance_covered"].max())],
        f"player_{player_2}_max_distance_covered": [safe_value(player_stats["player_2_distance_covered"].max())],
        f"player_{player_1}_max_rally_contribution": [safe_value(player_stats["player_1_rally_contribution"].max())],
        f"player_{player_2}_max_rally_contribution": [safe_value(player_stats["player_2_rally_contribution"].max())],
        f"player_{player_1}_total_shots": [safe_value(player_stats["player_1_total_shots"].max())],
        f"player_{player_2}_total_shots": [safe_value(player_stats["player_2_total_shots"].max())],
        f"player_{player_1}_max_rally_percentage": [safe_value(player_stats["player_1_rally_percentage"].max())],
        f"player_{player_2}_max_rally_percentage": [safe_value(player_stats["player_2_rally_percentage"].max())],
    }

    df = pd.DataFrame(max_stats)
    return df