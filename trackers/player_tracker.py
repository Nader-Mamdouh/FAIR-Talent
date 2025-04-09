import math
from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

# Define constants for minimum and maximum player area
MIN_PLAYER_AREA = 0  # Example value, adjust as needed
MAX_PLAYER_AREA = 50000  # Example value, adjust as needed
MIN_ASPECT_RATIO = 0.2  # Example value, adjust as needed
MAX_ASPECT_RATIO = 10.0  # Example value, adjust as needed

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
            """
            Chooses the two players based on multiple features:
            - Bounding box size
            - Aspect ratio
            - Position relative to court keypoints
            """
            if not player_dict:
                raise ValueError("Player dictionary is empty.")

            # List to store (track_id, score) pairs
            player_scores = []

            for track_id, bbox in player_dict.items():
                # Calculate bounding box features
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                aspect_ratio = bbox_height / bbox_width
                #print(f"bbox_area: {bbox_area}, aspect_ratio: {aspect_ratio}")
                # Filter based on bounding box size and aspect ratio
                if (bbox_area < MIN_PLAYER_AREA or bbox_area > MAX_PLAYER_AREA or
                    aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO):
                    continue  # Skip if the person doesn't meet size or aspect ratio criteria

                # Calculate position score (distance to court keypoints)
                player_center = get_center_of_bbox(bbox)
                min_distance = math.inf
                for i in range(0, len(court_keypoints), 2):
                    court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                    distance = measure_distance(player_center, court_keypoint)
                    min_distance = min(min_distance, distance)

                # Combine features into a score (lower distance and larger area are better)
                score = min_distance / bbox_area  # Adjust scoring formula as needed
                player_scores.append((track_id, score))

            # Check if there are at least two players
            if len(player_scores) < 2:
                raise ValueError("Not enough players to choose from.")

            # Sort by score (lower score is better)
            player_scores.sort(key=lambda x: x[1])

            # Choose the two players with the best scores
            chosen_players = [player_scores[0][0], player_scores[1][0]]
            return chosen_players    


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames