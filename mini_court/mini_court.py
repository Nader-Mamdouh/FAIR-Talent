import cv2
import numpy as np
import sys
from typing import Dict, List, Tuple

sys.path.append("../")

import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance,
)


class MiniCourt:
    def __init__(self, frame) -> None:
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        # canvas + court setup
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    # ----- geometry helpers -----

    def convert_meters_to_pixels(self, meters: float) -> float:
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width,
        )

    def set_canvas_background_box_position(self, frame) -> None:
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self) -> None:
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_court_drawing_key_points(self) -> None:
        # 14 keypoints * 2 coords = 28 entries
        drawing_key_points = [0] * 28

        # point 0: top-left of double court
        drawing_key_points[0] = int(self.court_start_x)
        drawing_key_points[1] = int(self.court_start_y)

        # point 1: top-right
        drawing_key_points[2] = int(self.court_end_x)
        drawing_key_points[3] = int(self.court_start_y)

        # point 2: bottom-left of doubles
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = (
            self.court_start_y
            + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        )

        # point 3: bottom-right of doubles
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]

        # point 4: left doubles alley top
        drawing_key_points[8] = (
            drawing_key_points[0]
            + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[9] = drawing_key_points[1]

        # point 5: left doubles alley bottom
        drawing_key_points[10] = (
            drawing_key_points[4]
            + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[11] = drawing_key_points[5]

        # point 6: right doubles alley top
        drawing_key_points[12] = (
            drawing_key_points[2]
            - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[13] = drawing_key_points[3]

        # point 7: right doubles alley bottom
        drawing_key_points[14] = (
            drawing_key_points[6]
            - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[15] = drawing_key_points[7]

        # point 8: back of no-man's land (near baseline)
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = (
            drawing_key_points[9]
            + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        )

        # point 9: singles sideline near baseline
        drawing_key_points[18] = (
            drawing_key_points[16]
            + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        )
        drawing_key_points[19] = drawing_key_points[17]

        # point 10: back of no-man's land (near service line)
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = (
            drawing_key_points[11]
            - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        )

        # point 11: singles sideline near service line
        drawing_key_points[22] = (
            drawing_key_points[20]
            + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        )
        drawing_key_points[23] = drawing_key_points[21]

        # point 12: center mark near baseline
        drawing_key_points[24] = int(
            (drawing_key_points[16] + drawing_key_points[18]) / 2
        )
        drawing_key_points[25] = drawing_key_points[17]

        # point 13: center service line
        drawing_key_points[26] = int(
            (drawing_key_points[20] + drawing_key_points[22]) / 2
        )
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self) -> None:
        # Pairs of keypoint indices (index in keypoints list, not xy)
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3),
        ]

    # ----- drawing -----

    def draw_court(self, frame) -> any:
        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw court lines
        for line in self.lines:
            start_idx, end_idx = line
            sx = int(self.drawing_key_points[start_idx * 2])
            sy = int(self.drawing_key_points[start_idx * 2 + 1])
            ex = int(self.drawing_key_points[end_idx * 2])
            ey = int(self.drawing_key_points[end_idx * 2 + 1])
            cv2.line(frame, (sx, sy), (ex, ey), (0, 0, 0), 2)

        # Draw net
        net_y = int(
            (self.drawing_key_points[1] + self.drawing_key_points[5]) / 2.0
        )
        net_start_point = (self.drawing_key_points[0], net_y)
        net_end_point = (self.drawing_key_points[2], net_y)
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame) -> any:
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED,
        )

        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames: List) -> List:
        output_frames: List = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    # ----- getters -----

    def get_start_point_of_mini_court(self) -> Tuple[int, int]:
        return int(self.court_start_x), int(self.court_start_y)

    def get_width_of_mini_court(self) -> float:
        return float(self.court_drawing_width)

    def get_court_drawing_keypoints(self) -> List[float]:
        return self.drawing_key_points

    # ----- coordinate transforms -----

    def get_mini_court_coordinates(
        self,
        object_position: Tuple[float, float],
        closest_key_point: Tuple[float, float],
        closest_key_point_index: int,
        player_height_in_pixels: float,
        player_height_in_meters: float,
    ) -> Tuple[float, float]:
        # Distance from keypoint in pixels
        dx_pixels, dy_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distances to meters using player height
        dx_m = convert_pixel_distance_to_meters(
            dx_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )
        dy_m = convert_pixel_distance_to_meters(
            dy_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )

        # Convert meters to mini-court pixels
        dx_mini = self.convert_meters_to_pixels(dx_m)
        dy_mini = self.convert_meters_to_pixels(dy_m)

        closest_mini_keypoint = (
            self.drawing_key_points[closest_key_point_index * 2],
            self.drawing_key_points[closest_key_point_index * 2 + 1],
        )

        mini_x = closest_mini_keypoint[0] + dx_mini
        mini_y = closest_mini_keypoint[1] + dy_mini

        return mini_x, mini_y

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes: List[Dict[int, List[float]]],
        ball_boxes: List[Dict[int, List[float]]],
        original_court_key_points: List[float],
        player_1: int,
        player_2: int,
    ) -> Tuple[List[Dict[int, Tuple[float, float]]], List[Dict[int, Tuple[float, float]]]]:
        player_heights_m = {
            player_1: 1.91,
            player_2: 1.93,
        }

        output_player_boxes: List[Dict[int, Tuple[float, float]]] = []
        output_ball_boxes: List[Dict[int, Tuple[float, float]]] = []

        for frame_num, player_bbox in enumerate(player_boxes):
            # Ball center in this frame
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)

            # Find player closest to ball (raw IDs)
            closest_player_id_to_ball = min(
                player_bbox.keys(),
                key=lambda pid: measure_distance(
                    ball_position, get_center_of_bbox(player_bbox[pid])
                ),
            )

            output_player_bboxes_dict: Dict[int, Tuple[float, float]] = {}

            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Closest court keypoint (in pixels)
                closest_idx = get_closest_keypoint_index(
                    foot_position, original_court_key_points, [0, 2, 12, 13]
                )
                closest_key_point = (
                    original_court_key_points[closest_idx * 2],
                    original_court_key_points[closest_idx * 2 + 1],
                )

                # Player height in pixels (max over a sliding window)
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [
                    get_height_of_bbox(player_boxes[i][player_id])
                    for i in range(frame_index_min, frame_index_max)
                    if player_id in player_boxes[i]
                ]

                if bboxes_heights_in_pixels:
                    max_player_height_in_pixels = max(bboxes_heights_in_pixels)
                else:
                    # Fallback to current bbox height if window is empty
                    max_player_height_in_pixels = get_height_of_bbox(bbox)

                mini_player_pos = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_idx,
                    max_player_height_in_pixels,
                    player_heights_m[player_id],
                )

                output_player_bboxes_dict[player_id] = mini_player_pos

                # If this player is closest to ball, map ball center to mini court as well
                if closest_player_id_to_ball == player_id:
                    ball_closest_idx = get_closest_keypoint_index(
                        ball_position, original_court_key_points, [0, 2, 12, 13]
                    )
                    ball_closest_key_point = (
                        original_court_key_points[ball_closest_idx * 2],
                        original_court_key_points[ball_closest_idx * 2 + 1],
                    )

                    mini_ball_pos = self.get_mini_court_coordinates(
                        ball_position,
                        ball_closest_key_point,
                        ball_closest_idx,
                        max_player_height_in_pixels,
                        player_heights_m[player_id],
                    )
                    output_ball_boxes.append({1: mini_ball_pos})

            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(
        self,
        frames: List,
        positions: List[Dict[int, Tuple[float, float]]],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> List:
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        return frames