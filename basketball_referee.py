import torch
import cv2
import argparse
from ultralytics import YOLO
import numpy as np
import os
import yaml
import shutil
from pathlib import Path
from collections import deque
import math
from sklearn.model_selection import train_test_split
import zipfile
import tempfile

# --- Constants ---
# Updated Class IDs to match dataset converter
# Based on the provided dataset.yaml:
#   0: player
#   1: hoop
#   2: ball
BALL_CLASS_ID = 2  # Basketball (was 1, now 2)
HOOP_CLASS_ID = 1  # Basketball hoop/rim (was 2, now 1)
PLAYER_CLASS_ID = 0  # Player (remains 0)

# Visual constants
HOOP_COLOR = (0, 0, 255)  # Red
BALL_COLOR = (0, 255, 0)  # Green
TRAJECTORY_COLOR = (255, 0, 255)  # Magenta
TEXT_COLOR = (255, 255, 255)  # White
SCORE_COLOR = (0, 255, 0)  # Green
MISS_COLOR = (0, 0, 255)  # Red
DEBUG_COLOR = (255, 255, 0)  # Yellow for debug info
ZONE_COLOR = (0, 255, 255)  # Cyan for scoring zone

# Detection parameters - VERY RELAXED FOR BETTER DETECTION
HOOP_CONFIDENCE_THRESHOLD = 0.2  # Even lower
BALL_CONFIDENCE_THRESHOLD = 0.1  # Very low for better detection
SCORE_DISPLAY_DURATION = 90
MISS_DISPLAY_DURATION = 60

# --- New Scoring Logic Constants ---
# Distance ball must be from player to be considered a shot (pixels)
MIN_BALL_PLAYER_SEPARATION_FOR_SHOT = 40  # Increased slightly for better separation
# Vertical velocity threshold for shot initiation (pixels/frame, negative means upward)
SHOT_INITIATION_VELOCITY_THRESHOLD = -2.5  # Made slightly stricter for upward motion
# Minimum consecutive frames the ball must show upward motion to be considered a shot
MIN_CONSECUTIVE_UPWARD_FRAMES = 3  # New constant for shot initiation robustness
# Offset from hoop's top-Y for the scoring plane (relative to hoop height)
SCORING_PLANE_Y_OFFSET_RATIO = 0.7  # Adjusted to 0.7 for more precise plane positioning
# Number of consecutive frames the ball must be fully below the scoring plane to confirm a score
MIN_FRAMES_BALL_THROUGH_PLANE = 2  # Reverted to 2 for more robustness
# Tolerance for horizontal alignment with hoop when passing through scoring plane (ratio of hoop width)
HOOP_HORIZONTAL_ALIGNMENT_TOLERANCE_RATIO = 0.8  # Increased for more leniency
# Maximum frames to predict ball/hoop position during brief occlusions
MAX_PREDICTION_FRAMES = 5
# Threshold for detecting a "rim bounce" (distance from hoop center)
RIM_BOUNCE_DETECTION_DISTANCE = 50  # pixels
# Minimum vertical distance ball must fall after peak to be considered a miss if not scored
MIN_FALL_DIST_FOR_MISS = 100
# Ratio of player height to consider ball too low for shot initiation (e.g., below waist)
PLAYER_WAIST_RATIO = 0.7  # Ball must be above the lower 30% of player's height


class CVATDatasetConverter:
    """Convert CVAT YOLO 1.1 annotations for free throw training with enhanced ball detection"""

    def __init__(self, cvat_paths, output_path):
        if isinstance(cvat_paths, (str, Path)):
            self.cvat_paths = [Path(cvat_paths)]
        else:
            self.cvat_paths = [Path(p) for p in cvat_paths]

        self.output_path = Path(output_path)
        self.class_mapping = {
            'player': 0,
            'person': 0,
            'ball': 1,
            'basketball': 1,
            'basket ball': 1,
            'hoop': 2,
            'rim': 2,
            'basket': 2,
        }
        self.all_images = []
        self.all_labels = []

    def extract_zip_if_needed(self, path):
        """Extract zip file if provided"""
        if path.suffix.lower() == '.zip':
            print(f"Extracting: {path}")
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return Path(temp_dir), temp_dir
        return path, None

    def convert_multiple_cvat_to_yolo(self):
        """Convert multiple CVAT datasets to YOLO format with verification"""
        print(f"Converting {len(self.cvat_paths)} CVAT dataset(s)...")

        # Create directory structure
        (self.output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        temp_dirs = []

        try:
            # Process each CVAT export
            for idx, cvat_path in enumerate(self.cvat_paths):
                print(f"\nProcessing dataset {idx + 1}/{len(self.cvat_paths)}: {cvat_path}")

                extracted_path, temp_dir = self.extract_zip_if_needed(cvat_path)
                if temp_dir:
                    temp_dirs.append(temp_dir)

                self.dataset_path = extracted_path
                self._collect_yolo11_format(dataset_index=idx)

            # Split and save
            self._split_and_save_combined_dataset()
            self._create_dataset_yaml()

            # Verify dataset quality
            self.verify_dataset()

            print(f"\nConversion complete! Total images: {len(self.all_images)}")

        finally:
            # Cleanup
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    def verify_dataset(self):
        """Verify dataset has proper ball annotations"""
        label_dir = self.output_path / 'labels' / 'train'
        ball_count = 0
        hoop_count = 0
        player_count = 0

        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    if class_id == BALL_CLASS_ID:
                        ball_count += 1
                    elif class_id == HOOP_CLASS_ID:
                        hoop_count += 1
                    elif class_id == PLAYER_CLASS_ID:
                        player_count += 1

        print(f"\nDataset Verification:")
        print(f"Ball annotations: {ball_count}")
        print(f"Hoop annotations: {hoop_count}")
        print(f"Player annotations: {player_count}")

        if ball_count < 100:
            print("Warning: Very few ball annotations - detection may be poor")
        elif ball_count < 500:
            print("Moderate number of ball annotations - consider adding more")
        else:
            print("Good number of ball annotations")

    def _collect_yolo11_format(self, dataset_index):
        """Collect images and labels from YOLO 1.1 format"""
        obj_train_data_path = self.dataset_path / 'obj_train_data'
        obj_names_path = self.dataset_path / 'obj.names'

        if not obj_train_data_path.exists():
            raise FileNotFoundError(f"obj_train_data not found in {self.dataset_path}")

        # Read class names
        if obj_names_path.exists():
            with open(obj_names_path, 'r') as f:
                for idx, line in enumerate(f):
                    class_name = line.strip().lower()
                    print(f"Class {idx}: {class_name}")

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(obj_train_data_path.glob(f'*{ext}')))

        print(f"Found {len(image_files)} images")

        # Collect images and labels
        for img_file in image_files:
            unique_name = f"dataset{dataset_index}_{img_file.name}"

            self.all_images.append({
                'src_path': img_file,
                'unique_name': unique_name,
                'dataset_index': dataset_index
            })

            label_file = obj_train_data_path / f"{img_file.stem}.txt"
            if label_file.exists():
                self.all_labels.append({
                    'src_path': label_file,
                    'unique_name': f"dataset{dataset_index}_{img_file.stem}.txt",
                    'dataset_index': dataset_index
                })

    def _split_and_save_combined_dataset(self):
        """Split dataset into train/val and save"""
        if not self.all_images:
            raise ValueError("No images found!")

        train_images, val_images = train_test_split(
            self.all_images, test_size=0.2, random_state=42
        )

        print(f"\nTrain: {len(train_images)} | Val: {len(val_images)}")

        # Save train set
        for img_info in train_images:
            self._save_image_and_label(img_info, 'train')

        # Save val set
        for img_info in val_images:
            self._save_image_and_label(img_info, 'val')

    def _save_image_and_label(self, img_info, split):
        """Save image and label files"""
        # Copy image
        dst_img = self.output_path / 'images' / split / img_info['unique_name']
        shutil.copy2(img_info['src_path'], dst_img)

        # Find and copy label
        for label_info in self.all_labels:
            if (label_info['dataset_index'] == img_info['dataset_index'] and
                    label_info['unique_name'].replace('.txt', '') ==
                    img_info['unique_name'].replace('.jpg', '').replace('.png', '')):
                dst_label = self.output_path / 'labels' / split / label_info['unique_name']
                shutil.copy2(label_info['src_path'], dst_label)
                break

    def _create_dataset_yaml(self):
        """
        Create dataset.yaml for training.
        The 'path' key in the YAML will be the absolute path of the 'output_path'
        provided by the user when running in 'convert' mode.
        """
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,  # Number of classes
            'names': {
                0: 'player',
                1: 'hoop',  # Updated to match user's dataset.yaml
                2: 'ball',  # Updated to match user's dataset.yaml
            }
        }

        with open(self.output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)


class FreeThrowModelTrainer:
    """Train YOLO model for free throw detection with enhanced ball detection"""

    def __init__(self, dataset_path, model_size='s'):
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None

    def train_model(self, epochs=150, batch_size=16, img_size=640, device='auto'):
        """Train the model with optimized parameters for ball detection"""
        print(f"Starting training with YOLOv8{self.model_size}...")

        # Auto-detect device
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        print(f"Using device: {device}")

        # Load base model
        self.model = YOLO(f'yolov8{self.model_size}.pt')

        # Training configuration with enhanced parameters for ball detection
        results = self.model.train(
            data=str(self.dataset_path / 'dataset.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project='freethrow_training',
            name=f'freethrow_yolov8{self.model_size}',
            patience=30,
            save_period=10,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            optimizer='AdamW',

            # Enhanced parameters for small object (ball) detection
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            degrees=2.0,
            translate=0.05,
            scale=0.3,
            fliplr=0.0,  # Keep orientation for ball detection
            mosaic=1.0,  # Higher mosaic for better context
            mixup=0.0,  # Disable mixup for clearer ball features
            copy_paste=0.0,  # Disable copy-paste augmentation

            # Small object specific
            overlap_mask=True,
            single_cls=False,
            # Augmentations that help with ball detection
            perspective=0.0005,
            shear=0.0,

            exist_ok=True,
            verbose=True,
        )

        print("Training completed!")
        return results

    def validate_model(self):
        """Validate the trained model with focus on ball detection metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        metrics = self.model.val()
        print(f"\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")

        # Print class-specific metrics if available
        if hasattr(metrics.box, 'ap_class'):
            print("\nClass-specific metrics:")
            for i, class_name in enumerate(['player', 'ball', 'hoop']):
                if i < len(metrics.box.ap_class):
                    print(f"{class_name}: AP50={metrics.box.ap50[i]:.3f}, AP50-95={metrics.box.ap[i]:.3f}")

        return metrics


class ShotTracker:
    """Track basketball shots with simple, reliable logic"""

    def __init__(self):
        self.positions = deque(maxlen=30)  # Reduced for faster response
        self.shot_phase = 'idle'  # idle, rising, peak, falling, potential_score, complete
        self.peak_height = None
        self.shot_start_y = None
        self.frames_near_hoop = 0
        self.last_position = None
        self.ball_passed_through = False  # True if ball passed through scoring plane
        self.frames_since_complete = 0
        self.min_shot_height = 10  # Very minimal height requirement
        self.frames_in_phase = 0
        self.max_frames_in_phase = 100  # Timeout for any phase

        # New attributes for improved scoring logic
        self.last_ball_bbox = None
        self.last_player_bbox = None  # Stores the bbox of the player closest to the ball
        self.ball_player_initial_distance = None  # Distance when shot initiated
        self.frames_ball_in_scoring_plane = 0
        self.ball_entered_hoop_area_from_above = False
        self.ball_exited_hoop_area_below = False
        self.rim_contact_detected = False
        self.last_hoop_center = None  # To help with prediction if hoop is lost
        self.last_hoop_bbox = None  # To help with prediction if hoop is lost
        self.debug_mode = False  # Initialize debug_mode here
        self.consecutive_upward_frames = 0  # New: for stricter shot initiation
        self.has_crossed_scoring_plane_once = False  # New: Tracks first crossing of scoring plane
        self.ball_fully_below_scoring_plane = False  # New: Tracks when entire ball is below scoring plane

    def ensure_attributes(self):
        """Ensure all required attributes exist for backward compatibility if needed"""
        if not hasattr(self, 'frames_in_phase'): self.frames_in_phase = 0
        if not hasattr(self, 'max_frames_in_phase'): self.max_frames_in_phase = 100
        if not hasattr(self, 'min_shot_height'): self.min_shot_height = 10
        if not hasattr(self, 'last_ball_bbox'): self.last_ball_bbox = None
        if not hasattr(self, 'last_player_bbox'): self.last_player_bbox = None
        if not hasattr(self, 'ball_player_initial_distance'): self.ball_player_initial_distance = None
        if not hasattr(self, 'frames_ball_in_scoring_plane'): self.frames_ball_in_scoring_plane = 0
        if not hasattr(self, 'ball_entered_hoop_area_from_above'): self.ball_entered_hoop_area_from_above = False
        if not hasattr(self, 'ball_exited_hoop_area_below'): self.ball_exited_hoop_area_below = False
        if not hasattr(self, 'rim_contact_detected'): self.rim_contact_detected = False
        if not hasattr(self, 'last_hoop_center'): self.last_hoop_center = None
        if not hasattr(self, 'last_hoop_bbox'): self.last_hoop_bbox = None
        if not hasattr(self, 'debug_mode'): self.debug_mode = False  # Ensure debug_mode exists
        if not hasattr(self, 'consecutive_upward_frames'): self.consecutive_upward_frames = 0
        if not hasattr(self, 'has_crossed_scoring_plane_once'): self.has_crossed_scoring_plane_once = False
        if not hasattr(self, 'ball_fully_below_scoring_plane'): self.ball_fully_below_scoring_plane = False

    def update(self, ball_info, hoop_info, player_bboxes, debug_mode_scorer=False):  # Add debug_mode_scorer parameter
        """
        Update shot tracking with new ball, hoop, and player positions.
        ball_info: {'center': (x,y), 'bbox': (x1,y1,x2,y2), 'confidence': float} or None
        hoop_info: {'center': (x,y), 'bbox': (x1,y1,w,h)} or None
        player_bboxes: list of (x1,y1,x2,y2) for all detected players
        debug_mode_scorer: boolean, debug mode status from the scorer
        """
        self.debug_mode = debug_mode_scorer  # Update debug_mode in ShotTracker
        self.ensure_attributes()

        ball_center = ball_info['center'] if ball_info else None
        ball_bbox = ball_info['bbox'] if ball_info else None
        hoop_center = hoop_info['center'] if hoop_info else self.last_hoop_center  # Use last known if current is None
        hoop_bbox = hoop_info['bbox'] if hoop_info else self.last_hoop_bbox  # Use last known if current is None

        # Update last known hoop position
        if hoop_info:
            self.last_hoop_center = hoop_info['center']
            self.last_hoop_bbox = hoop_info['bbox']

        self.frames_in_phase += 1

        # Auto-reset after shot completion
        if self.shot_phase == 'complete':
            self.frames_since_complete += 1
            if self.frames_since_complete > 15:  # Quick reset for next shot
                self.reset()
                return None

        # Timeout handling for active shot phases
        if self.shot_phase in ['rising', 'peak', 'falling', 'potential_score'] and \
                self.frames_in_phase > self.max_frames_in_phase:
            if self.debug_mode:
                print(f"DEBUG: Shot timeout in phase: {self.shot_phase} after {self.frames_in_phase} frames")
            self.shot_phase = 'complete'
            # If ball passed through, it's a score, otherwise a miss
            if self.ball_passed_through:
                if self.debug_mode: print("DEBUG: Timeout -> SCORE (ball passed through)")
                return 'score'
            if self.debug_mode: print("DEBUG: Timeout -> MISS (ball did not pass through)")
            return 'miss'

        # Handle ball disappearance (occlusion or out of frame)
        if ball_center is None:
            # If ball disappears during an active shot phase, try to predict or decide
            if self.shot_phase in ['rising', 'peak', 'falling', 'potential_score']:
                self.frames_since_complete += 1
                if self.debug_mode: print(f"DEBUG: Ball lost. Frames since complete: {self.frames_since_complete}")
                # If ball was just near hoop and disappeared, assume score
                if self.shot_phase == 'potential_score' and self.frames_since_complete < MAX_PREDICTION_FRAMES:
                    # Still in potential score, give it a few more frames
                    if self.debug_mode: print("DEBUG: Ball lost in potential_score, waiting for reappearance.")
                    return None
                elif self.shot_phase == 'potential_score' and self.ball_passed_through:
                    # Ball passed through and then disappeared
                    self.shot_phase = 'complete'
                    if self.debug_mode: print("DEBUG: Ball lost in potential_score after passing through -> SCORE")
                    return 'score'
                elif self.shot_phase == 'falling' and self.frames_near_hoop > 0 and self.frames_since_complete < MAX_PREDICTION_FRAMES:
                    # Ball was near hoop, give it a few frames to reappear or score
                    if self.debug_mode: print("DEBUG: Ball lost in falling phase near hoop, waiting.")
                    return None
                elif self.shot_phase == 'falling' and self.frames_near_hoop > 0 and self.ball_passed_through:
                    # Ball was near hoop, passed through, then disappeared
                    self.shot_phase = 'complete'
                    if self.debug_mode: print("DEBUG: Ball lost in falling phase after passing through -> SCORE")
                    return 'score'
                else:
                    # Ball disappeared for too long or not near hoop, consider it a miss
                    self.shot_phase = 'complete'
                    if self.debug_mode: print("DEBUG: Ball lost for too long/not near hoop -> MISS")
                    return 'miss'
            return None  # Not in an active shot phase, just lost track of ball

        # Store current ball and player info
        self.positions.append(ball_center)
        self.last_ball_bbox = ball_bbox
        # Find the player closest to the ball for shot initiation check
        if player_bboxes and ball_center:
            min_dist_to_player = float('inf')
            closest_player_bbox = None
            for p_bbox in player_bboxes:
                p_center = (int((p_bbox[0] + p_bbox[2]) / 2), int((p_bbox[1] + p_bbox[3]) / 2))
                dist = np.sqrt((ball_center[0] - p_center[0]) ** 2 + (ball_center[1] - p_center[1]) ** 2)
                if dist < min_dist_to_player:
                    min_dist_to_player = dist
                    closest_player_bbox = p_bbox
            self.last_player_bbox = closest_player_bbox
        else:
            self.last_player_bbox = None

        if len(self.positions) < 2:
            return None

        # Calculate vertical velocity (more responsive)
        recent_positions = list(self.positions)[-3:]
        vy = recent_positions[-1][1] - recent_positions[-2][1]  # Positive means falling, negative means rising

        # --- Shot Initiation Logic ---
        if self.shot_phase == 'idle':
            if ball_center and self.last_player_bbox:
                px1, py1, px2, py2 = self.last_player_bbox
                bx1, by1, bx2, by2 = ball_bbox
                ball_center_y = ball_center[1]
                player_center_x = (px1 + px2) / 2
                player_center_y = (py1 + py2) / 2
                player_height = py2 - py1

                # Condition 1: Ball moving significantly upward
                is_moving_up = vy < SHOT_INITIATION_VELOCITY_THRESHOLD

                # Condition 2: Ball is separated from the player's lower body/hands
                # This checks if the ball's center is above the lower 30% of the player's bounding box
                # or if the ball is horizontally outside the player's bounding box.
                ball_above_player_lower_body = ball_center_y < py2 - (player_height * 0.3)  # Above lower 30% of player
                ball_horizontally_separated = (bx1 > px2 or bx2 < px1)

                # Condition 3: Ball-player distance check (more lenient)
                ball_player_dist = np.sqrt(
                    (ball_center[0] - player_center_x) ** 2 + (ball_center[1] - player_center_y) ** 2)
                is_sufficiently_separated = ball_player_dist > MIN_BALL_PLAYER_SEPARATION_FOR_SHOT

                # Condition 4: Ball is not too low (e.g., below player's waist)
                ball_not_too_low = ball_center_y < py1 + (player_height * PLAYER_WAIST_RATIO)

                if is_moving_up and ball_not_too_low and (
                        ball_above_player_lower_body or ball_horizontally_separated or is_sufficiently_separated):
                    self.consecutive_upward_frames += 1
                    if self.debug_mode:
                        print(f"DEBUG: Potential upward frame. Consecutive: {self.consecutive_upward_frames}")
                    if self.consecutive_upward_frames >= MIN_CONSECUTIVE_UPWARD_FRAMES:
                        self.shot_phase = 'rising'
                        self.shot_start_y = ball_center[1]
                        self.peak_height = ball_center[1]
                        self.frames_in_phase = 0
                        self.ball_player_initial_distance = ball_player_dist
                        self.consecutive_upward_frames = 0  # Reset after shot initiated
                        if self.debug_mode:
                            print(
                                f"DEBUG: Shot detected! Starting Y: {self.shot_start_y}, Ball-Player Dist: {ball_player_dist:.1f}")
                            print(
                                f"  Conditions: Up={is_moving_up} (vy={vy:.2f}), AboveBody={ball_above_player_lower_body}, HorizSep={ball_horizontally_separated}, DistSep={is_sufficiently_separated}, NotTooLow={ball_not_too_low}")
                else:
                    self.consecutive_upward_frames = 0  # Reset if conditions not met
                    if self.debug_mode:
                        print(f"DEBUG: Shot NOT detected (idle):")
                        print(
                            f"  is_moving_up: {is_moving_up} (vy={vy:.2f} vs threshold {SHOT_INITIATION_VELOCITY_THRESHOLD})")
                        print(
                            f"  ball_above_player_lower_body: {ball_above_player_lower_body} (ball_y={ball_center_y}, player_lower_body_y={py2 - (player_height * 0.3):.2f})")
                        print(
                            f"  ball_horizontally_separated: {ball_horizontally_separated} (ball_x=[{bx1},{bx2}], player_x=[{px1},{px2}])")
                        print(
                            f"  is_sufficiently_separated: {is_sufficiently_separated} (dist={ball_player_dist:.1f} vs threshold {MIN_BALL_PLAYER_SEPARATION_FOR_SHOT})")
                        print(
                            f"  ball_not_too_low: {ball_not_too_low} (ball_y={ball_center_y}, player_waist_y={py1 + (player_height * PLAYER_WAIST_RATIO):.2f})")
            return None

        # --- Shot Phase Tracking ---
        elif self.shot_phase == 'rising':
            # Track peak
            if ball_center[1] < self.peak_height:
                self.peak_height = ball_center[1]

            # Detect when ball starts falling (or reaches peak)
            if vy >= 0.5 or self.frames_in_phase > 30:  # Small positive vy for falling, or timeout
                height_diff = self.shot_start_y - self.peak_height
                if height_diff > self.min_shot_height:  # Ensure it went up sufficiently
                    self.shot_phase = 'falling'
                    self.frames_in_phase = 0
                    if self.debug_mode:
                        print(f"DEBUG: Shot peak at Y: {self.peak_height} (height: {height_diff})")
                else:
                    # Very low shot, might be a pass or a very flat shot. Treat as falling.
                    self.shot_phase = 'falling'
                    self.frames_in_phase = 0
                    if self.debug_mode:
                        print(f"DEBUG: Low shot detected - continuing to track")

        elif self.shot_phase == 'falling':
            # Ensure we have hoop info for scoring checks
            if hoop_center and hoop_bbox and ball_bbox:
                hx, hy, hw, hh = hoop_bbox
                bx1, by1, bx2, by2 = ball_bbox
                ball_radius = (bx2 - bx1) / 2  # Approximate ball radius

                # Define the scoring plane just below the rim
                # This is the Y-coordinate where the *bottom* of the ball should pass
                scoring_plane_y = hy + hh * SCORING_PLANE_Y_OFFSET_RATIO

                # Check if ball is within the horizontal bounds of the hoop (with tolerance)
                hoop_horizontal_min = hx - hw * HOOP_HORIZONTAL_ALIGNMENT_TOLERANCE_RATIO / 2
                hoop_horizontal_max = hx + hw + hw * HOOP_HORIZONTAL_ALIGNMENT_TOLERANCE_RATIO / 2
                ball_center_x = (bx1 + bx2) / 2

                is_horizontally_aligned = (ball_center_x > hoop_horizontal_min and
                                           ball_center_x < hoop_horizontal_max)

                # Check if ball is entering the hoop area from above
                if not self.ball_entered_hoop_area_from_above:
                    if by2 < scoring_plane_y and by1 < hy + hh * 0.1:  # Ball top is above rim
                        # Ball is above scoring plane and generally above the hoop
                        self.ball_entered_hoop_area_from_above = True
                        self.frames_near_hoop = 0  # Reset counter for near hoop
                        if self.debug_mode:
                            print("DEBUG: Ball entered hoop area from above.")

                # Check for ball passing through the scoring plane (straddling it)
                if self.ball_entered_hoop_area_from_above and is_horizontally_aligned:
                    # If ball's bottom is below the plane and top is above (passing through)
                    if by2 > scoring_plane_y and by1 < scoring_plane_y:
                        self.frames_ball_in_scoring_plane += 1
                        self.frames_near_hoop += 1  # Still near hoop
                        self.has_crossed_scoring_plane_once = True  # Mark that it crossed at least once
                        if self.debug_mode:
                            print(
                                f"DEBUG: Ball straddling scoring plane. by1={by1:.1f}, by2={by2:.1f}, plane_y={scoring_plane_y:.1f}, frames_in_plane={self.frames_ball_in_scoring_plane}, is_horizontally_aligned={is_horizontally_aligned}")
                        if self.frames_ball_in_scoring_plane >= MIN_FRAMES_BALL_THROUGH_PLANE:
                            self.ball_passed_through = True
                            self.shot_phase = 'potential_score'
                            self.frames_in_phase = 0
                            if self.debug_mode:
                                print(
                                    f"DEBUG: Ball passed through scoring plane! Confirmed over {MIN_FRAMES_BALL_THROUGH_PLANE} frames.")
                    else:  # Ball is above the plane, but not straddling it (e.g., still rising, or moved away horizontally)
                        self.frames_ball_in_scoring_plane = 0  # Reset if not consistently passing through
                        if self.debug_mode:
                            print(
                                "DEBUG: Ball not consistently passing through scoring plane. Resetting frames_ball_in_scoring_plane.")

                # Check if entire ball is now below the scoring plane
                if by1 > scoring_plane_y:
                    self.ball_fully_below_scoring_plane = True
                    self.ball_exited_hoop_area_below = True  # This flag is redundant with ball_fully_below_scoring_plane, but kept for consistency
                    if self.debug_mode:
                        print(
                            f"DEBUG: Entire ball exited below scoring plane. by1={by1:.1f}, by2={by2:.1f}, plane_y={scoring_plane_y:.1f}")

                    # Crucial decision point: score or miss?
                    if self.ball_passed_through:  # If it successfully passed through the plane
                        self.shot_phase = 'complete'
                        if self.debug_mode: print("DEBUG: SCORE: Ball passed through and fully exited below.")
                        return 'score'
                    elif self.rim_contact_detected:  # If rim contact happened and it didn't pass through
                        self.shot_phase = 'complete'
                        if self.debug_mode: print(
                            "DEBUG: MISS: Rim contact without confirmed pass-through, then fully exited below.")
                        return 'miss'
                    elif self.has_crossed_scoring_plane_once:
                        # It straddled the plane at least once, but not for enough frames to confirm 'passed_through'
                        # and now it's fully below. This is likely a miss due to partial crossing or quick exit.
                        self.shot_phase = 'complete'
                        if self.debug_mode: print(
                            "DEBUG: MISS: Ball partially crossed but not confirmed, then fully exited below.")
                        return 'miss'
                    else:  # Ball just went below, no significant interaction with hoop area
                        self.shot_phase = 'complete'
                        if self.debug_mode: print(
                            "DEBUG: MISS: Ball fully exited below without entering hoop area or passing through.")
                        return 'miss'

                # --- Rim Contact Detection ---
                # If ball is very close to hoop center but not passing through
                dist_to_hoop_center = np.sqrt(
                    (ball_center[0] - hoop_center[0]) ** 2 + (ball_center[1] - hoop_center[1]) ** 2)
                if dist_to_hoop_center < RIM_BOUNCE_DETECTION_DISTANCE and not self.ball_passed_through:
                    # Check for sudden change in vertical velocity or horizontal deviation
                    # This is a heuristic for a bounce
                    if len(self.positions) >= 3:
                        prev_vy = self.positions[-2][1] - self.positions[-3][1]
                        current_vy = self.positions[-1][1] - self.positions[-2][1]
                        # If ball was falling, then suddenly slowed down or went up
                        if current_vy > prev_vy + 1.0:  # Ball slowed down or bounced up
                            self.rim_contact_detected = True
                            if self.debug_mode:
                                print("DEBUG: Rim contact detected!")

                # --- Miss Detection in Falling Phase (Early Exit Conditions) ---
                # 1. Ball moves significantly horizontally away from the hoop
                if hoop_center and abs(ball_center[0] - hoop_center[0]) > hw * 1.5:
                    if not self.ball_passed_through:
                        self.shot_phase = 'complete'
                        if self.debug_mode:
                            print("DEBUG: MISS: Ball moved too far horizontally from hoop.")
                        return 'miss'

                # 2. Ball falls significantly below the hoop without scoring
                if hoop_bbox and ball_center[1] > hy + hh * 3:  # Ball is far below hoop
                    if not self.ball_passed_through:
                        self.shot_phase = 'complete'
                        if self.debug_mode:
                            print("DEBUG: MISS: Ball fell too far below hoop.")
                        return 'miss'

            # If no hoop detected, use general falling heuristic as a fallback for miss
            elif ball_center and self.peak_height and (ball_center[1] - self.peak_height > MIN_FALL_DIST_FOR_MISS):
                # If ball has fallen a significant distance after peak and no hoop interaction
                self.shot_phase = 'complete'
                if self.debug_mode:
                    print("DEBUG: MISS: Ball fell significantly after peak without hoop (no hoop detected).")
                return 'miss'


        elif self.shot_phase == 'potential_score':
            if self.debug_mode:
                print(
                    f"DEBUG: Potential Score Phase. ball_passed_through={self.ball_passed_through}, ball_exited_hoop_area_below={self.ball_exited_hoop_area_below}, rim_contact_detected={self.rim_contact_detected}, ball_fully_below_scoring_plane={self.ball_fully_below_scoring_plane}")

            # If ball passed through and now fully exited below, it's a score
            if self.ball_passed_through and self.ball_fully_below_scoring_plane:
                self.shot_phase = 'complete'
                if self.debug_mode: print(
                    "DEBUG: SCORE: Confirmed in potential_score phase (passed through and fully exited).")
                return 'score'
            # If ball passed through but then moved away (e.g., bounced out)
            elif self.ball_passed_through and hoop_center and ball_center and \
                    abs(ball_center[0] - hoop_center[0]) > hoop_bbox[2] * 1.5:
                self.shot_phase = 'complete'
                if self.debug_mode: print(
                    "DEBUG: SCORE: Ball passed through, then moved away horizontally (counted as score).")
                return 'score'  # Still count as score if it passed through
            # If ball was in potential score but then clearly missed (e.g., bounced off rim and went far)
            elif self.rim_contact_detected and not self.ball_passed_through:
                self.shot_phase = 'complete'
                if self.debug_mode: print(
                    "DEBUG: MISS: Rim contact without confirmed pass-through in potential_score phase.")
                return 'miss'
            # If ball disappeared for too long in potential score phase without confirming
            elif ball_center is None and self.frames_in_phase > MAX_PREDICTION_FRAMES:
                self.shot_phase = 'complete'
                if self.debug_mode: print("DEBUG: MISS: Ball disappeared for too long in potential_score phase.")
                return 'miss'

        self.last_position = ball_center
        return None

    def reset(self):
        """Reset shot tracking"""
        self.shot_phase = 'idle'
        self.peak_height = None
        self.shot_start_y = None
        self.frames_near_hoop = 0
        self.ball_passed_through = False
        self.frames_since_complete = 0
        self.frames_in_phase = 0
        self.last_ball_bbox = None
        self.last_player_bbox = None
        self.ball_player_initial_distance = None
        self.frames_ball_in_scoring_plane = 0
        self.ball_entered_hoop_area_from_above = False
        self.ball_exited_hoop_area_below = False
        self.rim_contact_detected = False
        self.consecutive_upward_frames = 0  # Reset this too
        self.has_crossed_scoring_plane_once = False  # Reset this too
        self.ball_fully_below_scoring_plane = False  # Reset this too
        # Keep some position history for continuity
        if len(self.positions) > 5:
            self.positions = deque(list(self.positions)[-5:], maxlen=30)
        else:
            self.positions.clear()


class ImprovedFreeThrowScorer:
    """Improved free throw scoring with better shot detection"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.made_shots = 0
        self.missed_shots = 0
        self.shot_attempts = 0

        # Tracking
        self.shot_tracker = ShotTracker()
        self.hoop_bbox = None
        self.hoop_center = None
        self.hoop_history = deque(maxlen=10)  # Reduced for faster response

        # Ball tracking
        self.ball_history = deque(maxlen=5)  # Reduced for faster response
        self.last_ball_center = None
        self.last_ball_bbox = None  # Store bbox for shot tracker
        self.frames_without_ball = 0

        # Player tracking
        self.player_bboxes = []

        # Display
        self.score_display_counter = 0
        self.miss_display_counter = 0
        self.debug_mode = True  # This is the debug_mode for the scorer

        # Frame counter
        self.frame_count = 0
        self.last_score_frame = -30  # Reduced cooldown
        self.consecutive_shots = 0

    def detect_objects(self, frame):
        """Detect objects with YOLO"""
        results = self.model(frame,
                             conf=BALL_CONFIDENCE_THRESHOLD,  # Use general threshold for all, then filter
                             iou=0.3,
                             verbose=False,
                             imgsz=640)
        return results[0] if results else None

    def update_hoop_position(self, detections):
        """Update hoop position with smoothing and return hoop info"""
        hoop_found = False
        current_hoop_info = None

        if detections and detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()

            best_hoop = None
            best_conf = HOOP_CONFIDENCE_THRESHOLD

            for i, cls in enumerate(classes):
                if int(cls) == HOOP_CLASS_ID and confidences[i] > best_conf:
                    best_conf = confidences[i]
                    best_hoop = boxes[i].astype(int)

            if best_hoop is not None:
                x1, y1, x2, y2 = best_hoop
                self.hoop_history.append((x1, y1, x2 - x1, y2 - y1))  # Store (x, y, w, h)
                hoop_found = True

        # Average hoop position for stability
        if len(self.hoop_history) >= 2:
            avg_x = int(np.median([h[0] for h in self.hoop_history]))
            avg_y = int(np.median([h[1] for h in self.hoop_history]))
            avg_w = int(np.median([h[2] for h in self.hoop_history]))
            avg_h = int(np.median([h[3] for h in self.hoop_history]))

            self.hoop_bbox = (avg_x, avg_y, avg_w, avg_h)
            self.hoop_center = (avg_x + avg_w // 2, avg_y + avg_h // 2)
            current_hoop_info = {'center': self.hoop_center, 'bbox': self.hoop_bbox}
            return current_hoop_info

        # If no hoop found in current frame but we have history, use last known good
        if not hoop_found and self.hoop_bbox and self.hoop_center:
            return {'center': self.hoop_center, 'bbox': self.hoop_bbox}

        return None  # No hoop detected or in history

    def find_ball(self, detections):
        """Find basketball in detections and return ball info including bbox and confidence"""
        current_ball_info = None

        if detections and detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()

            balls = []
            for i, cls in enumerate(classes):
                if int(cls) == BALL_CLASS_ID:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    conf = confidences[i]
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    size = (x2 - x1) * (y2 - y1)

                    if 50 < size < 50000:  # Reasonable ball size range
                        balls.append({
                            'box': (x1, y1, x2, y2),
                            'center': center,
                            'confidence': conf,
                            'size': size
                        })

            if not balls:
                self.frames_without_ball += 1
                # Simple prediction for ball if lost for a few frames during active shot
                if self.frames_without_ball < MAX_PREDICTION_FRAMES and self.last_ball_center:
                    # Predict next position based on last known velocity
                    if len(self.ball_history) >= 2:
                        prev_center = self.ball_history[-2]
                        current_center = self.ball_history[-1]
                        vx = current_center[0] - prev_center[0]
                        vy = current_center[1] - prev_center[1]
                        predicted_center = (current_center[0] + vx, current_center[1] + vy)
                        # Use last known bbox size for prediction
                        if self.last_ball_bbox:
                            bx1, by1, bx2, by2 = self.last_ball_bbox
                            bw, bh = bx2 - bx1, by2 - by1
                            predicted_bbox = (int(predicted_center[0] - bw / 2), int(predicted_center[1] - bh / 2),
                                              int(predicted_center[0] + bw / 2), int(predicted_center[1] + bh / 2))
                            current_ball_info = {'center': predicted_center, 'bbox': predicted_bbox,
                                                 'confidence': 0.0}  # Add confidence 0.0 for predicted
                            # Do not add predicted position to history to avoid compounding errors
                            return current_ball_info
                return None

            # Find best ball
            best_ball = None
            if self.last_ball_center and self.frames_without_ball < MAX_PREDICTION_FRAMES:
                # Find ball closest to last position to maintain tracking
                min_dist = float('inf')
                for ball in balls:
                    dist = np.sqrt((ball['center'][0] - self.last_ball_center[0]) ** 2 +
                                   (ball['center'][1] - self.last_ball_center[1]) ** 2)
                    if dist < min_dist and dist < 150:  # Within reasonable distance
                        min_dist = dist
                        best_ball = ball

            if best_ball is None and balls:
                # If no close ball, take highest confidence ball
                best_ball = max(balls, key=lambda x: x['confidence'])

            if best_ball:
                self.frames_without_ball = 0
                self.ball_history.append(best_ball['center'])
                self.last_ball_center = best_ball['center']
                self.last_ball_bbox = best_ball['box']  # Store bbox

                # Smooth ball position using recent history
                if len(self.ball_history) >= 2:
                    avg_x = int(np.mean([b[0] for b in list(self.ball_history)[-2:]]))
                    avg_y = int(np.mean([b[1] for b in list(self.ball_history)[-2:]]))
                    best_ball['center'] = (avg_x, avg_y)
                    # Recalculate bbox based on smoothed center and original size
                    bx1, by1, bx2, by2 = best_ball['box']
                    bw, bh = bx2 - bx1, by2 - by1
                    best_ball['box'] = (int(avg_x - bw / 2), int(avg_y - bh / 2), int(avg_x + bw / 2),
                                        int(avg_y + bh / 2))

                current_ball_info = {'center': best_ball['center'], 'bbox': best_ball['box'],
                                     'confidence': best_ball['confidence']}
                return current_ball_info

        return None

    def find_players(self, detections):
        """Find player bounding boxes in detections"""
        player_bboxes = []
        if detections and detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()

            for i, cls in enumerate(classes):
                if int(cls) == PLAYER_CLASS_ID and confidences[i] > 0.3:  # Player confidence threshold
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    player_bboxes.append((x1, y1, x2, y2))
        return player_bboxes

    def process_video(self, source):
        """Process video for free throw scoring"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {source}")
            return

        print("\nImproved Free Throw Scorer Started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset scores")
        print("  'd' - Toggle debug mode")
        print("  'space' - Pause/Resume")
        print("  'm' - Manually mark current shot as MADE")
        print("  'n' - Manually mark current shot as MISSED")
        print("\nWatching for shots...")

        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1
            else:
                ret = True

            if ret:
                # Detect objects
                detections = self.detect_objects(frame)

                # Update hoop position
                hoop_info = self.update_hoop_position(detections)

                # Find ball and player detections
                ball_info = self.find_ball(detections)
                self.player_bboxes = self.find_players(detections)  # Update player bboxes

                # Update shot tracking with all relevant info, passing debug_mode
                old_phase = self.shot_tracker.shot_phase
                result = self.shot_tracker.update(ball_info, hoop_info, self.player_bboxes, self.debug_mode)

                # Count attempt when phase changes from idle to rising
                if old_phase == 'idle' and self.shot_tracker.shot_phase == 'rising':
                    self.shot_attempts += 1
                    print(f"\n🏀 Shot attempt #{self.shot_attempts} started!")

                # Handle scoring result
                if result == 'score':
                    self.made_shots += 1
                    self.score_display_counter = SCORE_DISPLAY_DURATION
                    self.last_score_frame = self.frame_count
                    self.consecutive_shots += 1
                    print(f"\n🏀 SCORE! Shot #{self.made_shots} made (Consecutive: {self.consecutive_shots})")
                    self.shot_tracker.reset()  # Always reset after result

                elif result == 'miss':
                    self.missed_shots += 1
                    self.miss_display_counter = MISS_DISPLAY_DURATION
                    self.consecutive_shots = 0
                    print(f"\n❌ MISS! Shot #{self.shot_attempts} missed (Total misses: {self.missed_shots})")
                    self.shot_tracker.reset()  # Always reset after result

                # Draw visualizations
                self.draw_frame(frame, ball_info, detections)

                cv2.imshow("Basketball Free Throw Scorer", frame)

            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.made_shots = 0
                self.missed_shots = 0
                self.shot_attempts = 0
                self.consecutive_shots = 0
                self.shot_tracker.reset()
                print("\nScores reset!")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                # Also update debug_mode in the shot_tracker instance
                self.shot_tracker.debug_mode = self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('m'):
                # Manual MADE
                if self.shot_tracker.shot_phase not in ['idle', 'complete']:
                    self.made_shots += 1
                    self.score_display_counter = SCORE_DISPLAY_DURATION
                    self.consecutive_shots += 1
                    print(f"\n🏀 MANUAL SCORE! Total: {self.made_shots}")
                    self.shot_tracker.reset()
                else:
                    print("No active shot to mark as made")
            elif key == ord('n'):
                # Manual MISS
                if self.shot_tracker.shot_phase not in ['idle', 'complete']:
                    self.missed_shots += 1
                    self.miss_display_counter = MISS_DISPLAY_DURATION
                    self.consecutive_shots = 0
                    print(f"\n❌ MANUAL MISS! Total: {self.missed_shots}")
                    self.shot_tracker.reset()
                else:
                    print("No active shot to mark as missed")

        cap.release()
        cv2.destroyAllWindows()

        # Handle any pending shot at video end
        if self.shot_tracker.shot_phase not in ['idle', 'complete']:
            print("\nVideo ended during a shot attempt")
            if self.shot_tracker.ball_passed_through:  # If it passed through, count as score
                self.made_shots += 1
                print("Counting final shot as MADE")
            else:
                self.missed_shots += 1
                print("Counting final shot as MISSED")

        self.print_final_stats()

    def draw_frame(self, frame, ball_info, detections):
        """Draw all visualizations on frame"""
        # Draw hoop
        if self.hoop_bbox:
            x, y, w, h = self.hoop_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), HOOP_COLOR, 3)
            cv2.circle(frame, self.hoop_center, 5, HOOP_COLOR, -1)

            # Show hoop detection quality
            if len(self.hoop_history) > 0:
                confidence = len(self.hoop_history) / 10.0
                cv2.putText(frame, f"Hoop: {confidence:.0%}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, HOOP_COLOR, 1)

            # Draw scoring zone in debug mode
            if self.debug_mode:
                # Entry zone (more generous)
                zone_x1 = x - int(w * 0.5)
                zone_x2 = x + w + int(w * 0.5)
                zone_y1 = y - h
                zone_y2 = y + int(h * 3)  # Extends further down

                cv2.rectangle(frame,
                              (zone_x1, zone_y1),
                              (zone_x2, zone_y2),
                              ZONE_COLOR, 1)
                cv2.putText(frame, "SCORING ZONE", (zone_x1, zone_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOR, 1)

                # Draw precise scoring plane line
                scoring_plane_y = y + h * SCORING_PLANE_Y_OFFSET_RATIO
                cv2.line(frame, (x, int(scoring_plane_y)), (x + w, int(scoring_plane_y)), (255, 255, 0), 2)
                cv2.putText(frame, "SCORING PLANE", (x + w + 5, int(scoring_plane_y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Draw rim line (top of the hoop)
                cv2.line(frame, (x, y), (x + w, y), (255, 255, 0), 2)
                cv2.putText(frame, "RIM TOP", (x + w + 5, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw ball
        if ball_info:
            x1, y1, x2, y2 = ball_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
            cv2.circle(frame, ball_info['center'], 8, BALL_COLOR, -1)

            # Show ball tracking status
            track_color = (0, 255, 0) if self.frames_without_ball == 0 else (0, 165, 255)
            cv2.putText(frame, f"Ball: {ball_info['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, track_color, 1)

        # Draw trajectory
        if len(self.shot_tracker.positions) > 2:
            positions = list(self.shot_tracker.positions)
            for i in range(1, len(positions)):
                # Convert positions to integer tuples
                pt1 = tuple(map(int, positions[i - 1]))
                pt2 = tuple(map(int, positions[i]))
                thickness = max(1, 3 - (len(positions) - i) // 10)
                cv2.line(frame, pt1, pt2, TRAJECTORY_COLOR, thickness)

        # Draw all detected objects in debug mode
        if self.debug_mode and detections and detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()

            for i, cls in enumerate(classes):
                if confidences[i] > 0.1:  # Show low confidence detections too
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    class_name = ['player', 'hoop', 'ball'][int(cls)]  # Updated class names for drawing
                    color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)][int(cls)]  # Updated colors for drawing
                    cv2.putText(frame, f"{class_name}: {confidences[i]:.2f}",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, color, 1)

        # Draw detected players
        for p_bbox in self.player_bboxes:
            x1, y1, x2, y2 = p_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for players
            cv2.putText(frame, "Player", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw stats
        self.draw_stats(frame)

        # Draw detection quality indicator
        quality_y = 150  # Moved down to avoid overlap
        if not self.hoop_bbox:
            cv2.putText(frame, "⚠️ NO HOOP DETECTED - Using position heuristics", (380, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.frames_without_ball > 0:  # Only show if ball is currently missing
            cv2.putText(frame, f"⚠️ BALL LOST ({self.frames_without_ball} frames)", (380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Draw shot phase indicator
        if self.shot_tracker.shot_phase != 'idle':
            phase_text = f"SHOT IN PROGRESS: {self.shot_tracker.shot_phase.upper()}"
            color = (0, 255, 255)  # Cyan by default

            if self.shot_tracker.shot_phase == 'potential_score':
                color = (0, 255, 0)  # Green for potential score
            elif self.shot_tracker.shot_phase == 'complete':
                color = (255, 255, 0)  # Yellow for complete

            # Add urgency indicator
            frames_in_phase = getattr(self.shot_tracker, 'frames_in_phase', 0)
            max_frames = getattr(self.shot_tracker, 'max_frames_in_phase', 100)

            if frames_in_phase > 50:
                phase_text += " (TIMEOUT SOON!)"
                color = (0, 165, 255)  # Orange

            cv2.putText(frame, phase_text,
                        (frame.shape[1] // 2 - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show progress bar
            progress = min(frames_in_phase / max_frames, 1.0)
            bar_width = 200
            bar_x = frame.shape[1] // 2 - bar_width // 2
            bar_y = 45
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 10), color, -1)

        # Draw debug info
        if self.debug_mode:
            # Ensure frames_in_phase exists
            frames_in_phase = getattr(self.shot_tracker, 'frames_in_phase', 0)

            debug_info = [
                f"Shot Phase: {self.shot_tracker.shot_phase}",
                f"Frames in Phase: {frames_in_phase}",
                f"Frames Near Hoop: {self.shot_tracker.frames_near_hoop}",
                f"Ball Passed Through: {self.shot_tracker.ball_passed_through}",
                f"Ball Entered Hoop Area: {self.shot_tracker.ball_entered_hoop_area_from_above}",
                f"Ball Exited Hoop Area Below: {self.shot_tracker.ball_exited_hoop_area_below}",
                f"Rim Contact: {self.shot_tracker.rim_contact_detected}",
                f"Frames in Scoring Plane: {self.shot_tracker.frames_ball_in_scoring_plane}",
                f"Frame: {self.frame_count}",
                f"Consecutive Shots: {self.consecutive_shots}",
                f"Frames Without Ball: {self.frames_without_ball}",
                f"Hoop Detected: {self.hoop_bbox is not None}",
                f"Consec. Upward Frames: {self.shot_tracker.consecutive_upward_frames}",  # New debug info
                f"Has Crossed Plane Once: {self.shot_tracker.has_crossed_scoring_plane_once}",  # New debug info
                f"Ball Fully Below Plane: {self.shot_tracker.ball_fully_below_scoring_plane}"  # New debug info
            ]

            y_offset = 150
            for info in debug_info:
                cv2.putText(frame, info, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEBUG_COLOR, 1)
                y_offset += 25

            # Show shot requirements
            if self.shot_tracker.shot_phase != 'idle':
                req_text = "Shot Requirements: Up > Peak > Down > Through Scoring Plane > Exit Below"
                cv2.putText(frame, req_text, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Show why shot might fail
                if self.shot_tracker.shot_phase == 'falling' and not self.hoop_bbox:
                    cv2.putText(frame, "WARNING: No hoop detected!", (10, frame.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw notifications
        if self.score_display_counter > 0:
            self.draw_notification(frame, "SCORED!", SCORE_COLOR,
                                   1.0 - (SCORE_DISPLAY_DURATION - self.score_display_counter) / 30)
            self.score_display_counter -= 1
        elif self.miss_display_counter > 0:
            self.draw_notification(frame, "MISSED!", MISS_COLOR,
                                   1.0 - (MISS_DISPLAY_DURATION - self.miss_display_counter) / 30)
            self.miss_display_counter -= 1

    def draw_stats(self, frame):
        """Draw statistics"""
        total = self.made_shots + self.missed_shots
        accuracy = (self.made_shots / total * 100) if total > 0 else 0

        # Background (expanded)
        cv2.rectangle(frame, (10, 10), (380, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (380, 130), (255, 255, 255), 2)

        # Stats
        cv2.putText(frame, f"Made: {self.made_shots}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, SCORE_COLOR, 2)
        cv2.putText(frame, f"Missed: {self.missed_shots}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, MISS_COLOR, 2)
        cv2.putText(frame, f"Attempts: {self.shot_attempts} | Accuracy: {accuracy:.1f}%", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"Consecutive: {self.consecutive_shots}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    def draw_notification(self, frame, text, color, alpha=1.0):
        """Draw notification with fade effect"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 2.5 + alpha * 0.5
        thickness = int(3 * alpha)

        if thickness < 1:
            return

        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - text_w) // 2
        y = int(h // 2 - (1 - alpha) * 50)

        # Shadow
        cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1)
        # Text
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)

    def print_final_stats(self):
        """Print final statistics"""
        total = self.made_shots + self.missed_shots
        accuracy = (self.made_shots / total * 100) if total > 0 else 0
        attempt_accuracy = (self.made_shots / self.shot_attempts * 100) if self.shot_attempts > 0 else 0

        print("\n" + "=" * 50)
        print("FINAL FREE THROW STATISTICS")
        print("=" * 50)
        print(f"Shot Attempts:     {self.shot_attempts}")
        print(f"Shots Made:        {self.made_shots}")
        print(f"Shots Missed:      {self.missed_shots}")
        print(f"Total Scored:      {total}")
        print(f"Scoring Accuracy:  {accuracy:.1f}%")
        print(f"Attempt Accuracy:  {attempt_accuracy:.1f}%")
        if self.shot_attempts > total:
            print(f"Unresolved Shots:  {self.shot_attempts - total}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Basketball Free Throw Scorer")
    parser.add_argument("--mode", choices=['convert', 'train', 'inference'], required=True,
                        help="Mode: convert data, train model, or run inference")

    # Conversion arguments
    parser.add_argument("--cvat_paths", nargs='+',
                        help="Path(s) to CVAT YOLO 1.1 exports")
    parser.add_argument("--output_path", help="Output path for dataset")

    # Training arguments
    parser.add_argument("--dataset_path", help="Path to dataset")
    parser.add_argument("--model_size", choices=['n', 's', 'm', 'l'], default='s',
                        help="Model size (default: s)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default='auto')

    # Inference arguments
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--video_paths", nargs='*', help="Video files")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")

    args = parser.parse_args()

    if args.mode == 'convert':
        if not args.cvat_paths or not args.output_path:
            print("Error: --cvat_paths and --output_path required")
            return

        converter = CVATDatasetConverter(args.cvat_paths, args.output_path)
        converter.convert_multiple_cvat_to_yolo()

    elif args.mode == 'train':
        if not args.dataset_path:
            print("Error: --dataset_path required")
            return

        trainer = FreeThrowModelTrainer(args.dataset_path, args.model_size)
        trainer.train_model(args.epochs, args.batch_size, device=args.device)
        trainer.validate_model()

        print(f"\nModel saved to: freethrow_training/freethrow_yolov8{args.model_size}/weights/best.pt")

    elif args.mode == 'inference':
        if not args.model_path:
            print("Error: --model_path required")
            return

        scorer = ImprovedFreeThrowScorer(args.model_path)

        if args.webcam:
            print("Starting webcam mode...")
            scorer.process_video(0)
        elif args.video_paths:
            for path in args.video_paths:
                if os.path.exists(path):
                    print(f"\nProcessing: {path}")
                    scorer = ImprovedFreeThrowScorer(args.model_path)
                    scorer.process_video(path)
                else:
                    print(f"File not found: {path}")
        else:
            print("No input provided. Use --webcam or --video_paths")


if __name__ == "__main__":
    main()
