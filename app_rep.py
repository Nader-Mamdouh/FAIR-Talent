import pandas as pd

def calculate_player_scores(data):
    df = pd.DataFrame(data)

    # Player 1: strip the "player_1_" prefix from keys
    player_1_data = {
        col.replace("player_1_", ""): float(df[col].iloc[0])
        for col in df.columns
        if col.startswith("player_1_")
    }

    # Player 2: strip the "player_2_" prefix from keys
    player_2_data = {
        col.replace("player_2_", ""): float(df[col].iloc[0])
        for col in df.columns
        if col.startswith("player_2_")
    }

    return {
        "status": "success",
        "data": {
            "player_1": player_1_data,
            "player_2": player_2_data,
        },
    }