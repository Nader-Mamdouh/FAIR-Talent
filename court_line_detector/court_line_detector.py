import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from typing import List, Tuple, Optional


class CourtLineDetector:
    def __init__(self, model_path: str) -> None:
        # Base ResNet50 backbone
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)

        # Load trained weights
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Preprocessing pipeline: BGR → RGB, resize, normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict 14 keypoints (x,y) on the court from a BGR image.
        Returns a flat array of length 28: [x0, y0, x1, y1, ..., x13, y13]
        in original image coordinates.
        """
        # Convert BGR→RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        # Scale back from 224×224 to original size
        keypoints[::2] *= original_w / 224.0  # x coords
        keypoints[1::2] *= original_h / 224.0  # y coords

        return keypoints

    def get_court_bounds(self, keypoints: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Extract the four main court corners from the 14 keypoints.
        Expects keypoints of length 28.
        """
        if keypoints.shape[0] != 28:
            print("Error: Invalid number of keypoints detected (expected 28).")
            return None

        # Using your original indices (0,1,3,2)
        court_corners = [
            (keypoints[0], keypoints[1]),   # point 0
            (keypoints[2], keypoints[3]),   # point 1
            (keypoints[6], keypoints[7]),   # point 3
            (keypoints[4], keypoints[5]),   # point 2
        ]

        return court_corners

    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw keypoints and their indices on the image.
        """
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(
                image,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_court_boundaries(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw the main court polygon on the image using the corner keypoints.
        """
        court_corners = self.get_court_bounds(keypoints)
        if court_corners is None:
            return image

        pts = np.array(court_corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return image

    def draw_keypoints_on_video(self, video_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        For each frame in a video, predict keypoints and draw them + court boundaries.
        """
        output_video_frames: List[np.ndarray] = []

        for frame in video_frames:
            keypoints = self.predict(frame)
            frame = self.draw_keypoints(frame, keypoints)
            frame = self.draw_court_boundaries(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames