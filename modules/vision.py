# modules/vision.py
"""
Visual Processing Module for Narrative Scene Understanding.
This module handles object segmentation, visual description, person tracking, and face recognition.
"""

import os
import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import time

class VisualProcessor:
    """
    Processes visual frames for narrative scene understanding.
    Handles segmentation, captioning, tracking, and facial recognition.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visual processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_segmentation_model()
        self._init_caption_model()
        self._init_tracker()
        self._init_face_recognition()
        
        # Object detection cache for efficiency
        self.detection_cache = {}
    
    def _init_segmentation_model(self):
        """Initialize the segmentation model (SAM)."""
        self.logger.info("Initializing segmentation model...")
        
        try:
            from segment_anything import SamPredictor, sam_model_registry
            
            sam_checkpoint = self.config.get("model_paths", {}).get(
                "sam", "models/sam_vit_h_4b8939.pth")
            
            model_type = "vit_h"
            if "vit_l" in sam_checkpoint:
                model_type = "vit_l"
            elif "vit_b" in sam_checkpoint:
                model_type = "vit_b"
            
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            
            self.sam_predictor = SamPredictor(sam)
            self.logger.info(f"SAM model loaded: {model_type}")
        except ImportError:
            self.logger.warning("Segment Anything not available, falling back to YOLOv5 for detection")
            self.sam_predictor = None
            
            # Load YOLOv5 as fallback
            try:
                import torch.hub
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.to(self.device)
                self.logger.info("YOLOv5 model loaded as fallback")
            except Exception as e:
                self.logger.error(f"Failed to load YOLOv5: {e}")
                self.yolo_model = None
    
    def _init_caption_model(self):
        """Initialize the image captioning model (BLIP-2)."""
        self.logger.info("Initializing caption model...")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
            model.to(self.device)
            
            self.caption_processor = processor
            self.caption_model = model
            self.logger.info("BLIP caption model loaded")
        except ImportError:
            self.logger.warning("BLIP not available, visual descriptions will be limited")
            self.caption_processor = None
            self.caption_model = None
    
    def _init_tracker(self):
        """Initialize the object tracking model (DeepSORT)."""
        self.logger.info("Initializing object tracker...")
        
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            self.tracker = DeepSort(
                max_age=30,                          # Maximum frames to keep track of missing objects
                n_init=3,                           # Number of frames before confirming a track
                max_cosine_distance=0.3,            # Threshold for cosine distance in appearance matching
                nn_budget=100,                      # Maximum size of the appearance history
                override_track_class=None,          # Custom track class
                embedder="mobilenet",              # Feature extractor
                half=self.device.type == "cuda",    # Use half precision for speed with CUDA
                bgr=True,                           # Input format
                embedder_gpu=self.device.type == "cuda",
                embedder_model_name=None,           # Use default model for feature extraction
                embedder_wts=None,                  # Use default weights
                polygon=False,                      # Use rectangle format
                today=None                          # Used for MOT Challenge evaluation
            )
            self.logger.info("DeepSORT tracker initialized")
        except ImportError:
            self.logger.warning("DeepSORT not available, using simple tracker")
            self.tracker = SimpleTracker()
    
    def _init_face_recognition(self):
        """Initialize the face recognition model (InsightFace)."""
        self.logger.info("Initializing face recognition...")
        
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.face_app = face_app
            self.logger.info("InsightFace model loaded")
        except ImportError:
            self.logger.warning("InsightFace not available, character recognition will be limited")
            self.face_app = None
    
    def process_frames(self, frames: List[Tuple[float, np.ndarray]]) -> List[Dict]:
        """
        Process a list of video frames.
        
        Args:
            frames: List of (timestamp, frame) tuples
            
        Returns:
            List of processed frame data dictionaries
        """
        self.logger.info(f"Processing {len(frames)} frames...")
        
        results = []
        for frame_idx, (timestamp, frame) in enumerate(frames):
            self.logger.debug(f"Processing frame {frame_idx} at timestamp {timestamp:.2f}s")
            
            # Process frame
            frame_data = self._process_single_frame(timestamp, frame, frame_idx)
            results.append(frame_data)
            
            # Status update for long videos
            if (frame_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {frame_idx + 1}/{len(frames)} frames")
        
        self.logger.info("Frame processing complete")
        return results
    
    def _process_single_frame(self, timestamp: float, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process a single video frame.
        
        Args:
            timestamp: Frame timestamp in seconds
            frame: NumPy array containing the frame image
            frame_idx: Index of the frame in the sequence
            
        Returns:
            Dictionary containing frame analysis data
        """
        # Detect objects and generate segmentation masks
        objects = self._detect_objects(frame, frame_idx)
        
        # Generate frame-level caption
        frame_caption = self._generate_frame_caption(frame)
        
        # Track objects across frames
        tracked_objects = self._track_objects(objects, frame)
        
        # Perform face recognition
        recognized_faces = self._recognize_faces(frame, objects)
        
        # Detect actions
        actions = self._detect_actions(frame, objects, tracked_objects)
        
        # Apply face recognition to person objects
        self._apply_face_recognition(objects, recognized_faces)
        
        # Create frame data dictionary
        frame_data = {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "overall_caption": frame_caption,
            "objects": objects,
            "actions": actions,
            "faces": recognized_faces
        }
        
        return frame_data
    
    def _detect_objects(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Detect and segment objects in a frame.
        
        Args:
            frame: Frame image
            frame_idx: Index of the frame
            
        Returns:
            List of detected objects with their properties
        """
        objects = []
        
        if self.sam_predictor:
            # Use Segment Anything Model (SAM)
            try:
                # Set image for the predictor
                self.sam_predictor.set_image(frame)
                
                # Generate automatic masks
                masks, scores, logits = self.sam_predictor.predict()
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score < 0.5:
                        continue  # Skip low-confidence masks
                    
                    # Convert mask to bounding box
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue
                    
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    
                    # Skip tiny objects
                    if (x_max - x_min) < 20 or (y_max - y_min) < 20:
                        continue
                    
                    # Extract object image
                    obj_img = frame.copy()
                    obj_img[~mask] = [0, 0, 0]  # Set background to black
                    obj_img = obj_img[y_min:y_max, x_min:x_max]
                    
                    # Determine object type (person vs. object)
                    # For SAM we need to use the caption model to classify
                    caption = self._generate_object_caption(obj_img)
                    
                    is_person = any(keyword in caption.lower() 
                                  for keyword in ["person", "man", "woman", "boy", "girl", "child", "human"])
                    
                    obj_type = "person" if is_person else "object"
                    
                    # Create object dictionary
                    obj = {
                        "id": f"{frame_idx}_{i}",
                        "type": obj_type,
                        "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "score": float(score),
                        "mask": mask,
                        "caption": caption,
                        "track_id": None  # Will be set by tracker
                    }
                    
                    objects.append(obj)
            
            except Exception as e:
                self.logger.error(f"Error in SAM processing: {e}")
                # Fall back to YOLO if SAM fails
                objects = self._detect_with_yolo(frame, frame_idx)
        
        else:
            # Use YOLOv5 as fallback
            objects = self._detect_with_yolo(frame, frame_idx)

        for obj in objects:
            if obj['type'] in ['car', 'vehicle', 'truck']:
                obj['color'] = self._detect_object_color(frame, obj['box'])

        return objects

    def _detect_object_color(self, frame: np.ndarray, box: List[int]) -> str:
        """Detect the dominant color of an object."""
        x_min, y_min, x_max, y_max = box
        
        # Extract object region
        obj_region = frame[y_min:y_max, x_min:x_max]
        
        # Convert to RGB for better color analysis
        obj_rgb = cv2.cvtColor(obj_region, cv2.COLOR_BGR2RGB)
        
        # Reshape for clustering
        pixels = obj_rgb.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), n_colors, None, criteria, 10, flags)
        
        # Count labels to find the most dominant color
        counts = np.bincount(labels.flatten())
        
        # Get the dominant color
        dominant_color = centers[np.argmax(counts)]
        
        # Map RGB values to color names
        color = self._map_rgb_to_color(dominant_color)
        
        return color

    def _map_rgb_to_color(self, rgb: np.ndarray) -> str:
        """Map RGB values to common color names."""
        # Define color ranges for common colors
        color_ranges = {
            'red': [(0, 0, 100), (80, 80, 255)],
            'blue': [(100, 0, 0), (255, 80, 80)],
            'green': [(0, 100, 0), (80, 255, 80)],
            'yellow': [(0, 100, 100), (80, 255, 255)],
            'black': [(0, 0, 0), (50, 50, 50)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'gray': [(50, 50, 50), (200, 200, 200)]
        }
        
        # Check which color range the RGB value falls into
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            if np.all(rgb >= lower) and np.all(rgb <= upper):
                return color_name
        
        return 'unknown'
    
    def _detect_with_yolo(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Detect objects using YOLOv5 (fallback method).
        
        Args:
            frame: Frame image
            frame_idx: Index of the frame
            
        Returns:
            List of detected objects with their properties
        """
        if self.yolo_model is None:
            return []
        
        objects = []
        
        try:
            # Run inference
            results = self.yolo_model(frame)
            
            # Process detections
            for i, (x1, y1, x2, y2, conf, cls) in enumerate(results.xyxy[0].cpu().numpy()):
                if conf < 0.5:
                    continue  # Skip low confidence detections
                
                # Get class name
                class_name = self.yolo_model.names[int(cls)]
                
                # Extract object image
                obj_img = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Generate caption for the object
                caption = f"{class_name}"
                if self.caption_model:
                    detailed_caption = self._generate_object_caption(obj_img)
                    if detailed_caption:
                        caption = detailed_caption
                
                # Create object dictionary
                obj = {
                    "id": f"{frame_idx}_{i}",
                    "type": "person" if class_name in ["person"] else class_name,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(conf),
                    "caption": caption,
                    "track_id": None  # Will be set by tracker
                }
                
                objects.append(obj)
                
        except Exception as e:
            self.logger.error(f"Error in YOLO detection: {e}")
        
        return objects
    
    def _generate_frame_caption(self, frame: np.ndarray) -> str:
        """
        Generate a caption for the entire frame.
        
        Args:
            frame: Frame image
            
        Returns:
            Caption string
        """
        if self.caption_model is None:
            return ""
        
        try:
            # Resize image if too large
            h, w = frame.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Convert to PIL image for the model
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Generate caption
            inputs = self.caption_processor(pil_image, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_length=80)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        
        except Exception as e:
            self.logger.error(f"Error generating frame caption: {e}")
            return ""
    
    def _generate_object_caption(self, obj_img: np.ndarray, obj_type: str = None, color: str = None) -> str:
        """
        Generate a caption for an object region.
        
        Args:
            obj_img: Object image region
            obj_type: Type of object if known
            color: Detected color if available
            
        Returns:
            Caption string with enhanced description
        """
        if self.caption_model is None or obj_img.size == 0:
            return ""
        
        try:
            # Convert to PIL image for the model
            pil_image = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
            
            # Generate caption
            inputs = self.caption_processor(pil_image, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Enhance caption with detected properties
            if obj_type and color and obj_type in ['car', 'vehicle', 'truck']:
                # Check if color is already in caption
                if color not in caption.lower():
                    # Add color to caption
                    caption = caption.replace(obj_type, f"{color} {obj_type}")
            
            return caption
            
        except Exception as e:
            self.logger.error(f"Error generating object caption: {e}")
            return ""
        
    def _track_objects(self, objects: List[Dict], frame: np.ndarray) -> List:
        """
        Track objects across frames.
        
        Args:
            objects: List of detected objects
            frame: Current frame
            
        Returns:
            List of tracked objects
        """
        if len(objects) == 0:
            return []
        
        try:
            # Convert objects to format required by tracker
            detections = []
            for obj in objects:
                box = obj["box"]
                score = obj["score"]
                label = obj["type"]
                
                # Format: [x1, y1, x2, y2, score, class_id]
                detection = [box[0], box[1], box[2], box[3], score, 0 if label == "person" else 1]
                detections.append(detection)
            
            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)
            # Add this check after the above line:
            if not isinstance(tracks, list) and not hasattr(tracks, '__iter__'):
                self.logger.error(f"Tracker returned non-iterable result: {tracks}")
                return []
            
            # Update object track_ids
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()  # Left, Top, Right, Bottom coordinates
                
                # Find the closest object to this track
                best_match = None
                best_iou = 0
                
                for obj in objects:
                    box = obj["box"]
                    iou = self._calculate_iou(
                        [ltrb[0], ltrb[1], ltrb[2], ltrb[3]], 
                        box
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj
                
                # Update the object with this track_id if it's a good match
                if best_match is not None and best_iou > 0.5:
                    best_match["track_id"] = track_id
            
            return tracks
        
        except Exception as e:
            self.logger.error(f"Error in object tracking: {e}")
            return []
    
    def _recognize_faces(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """
        Perform face recognition on persons in the frame.
        
        Args:
            frame: Current frame
            objects: List of detected objects
            
        Returns:
            List of recognized faces
        """
        if self.face_app is None:
            return []
        
        try:
            # Detect faces
            faces = self.face_app.get(frame)
            
            recognized_faces = []
            for i, face in enumerate(faces):
                box = face.bbox.astype(int)
                embedding = face.embedding
                
                # Create face dictionary
                face_data = {
                    "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "score": float(face.det_score),
                    "embedding": embedding,
                    "gender": face.gender,
                    "age": int(face.age),
                    "person_id": None  # Will be filled later by identity matching
                }
                
                recognized_faces.append(face_data)
            
            return recognized_faces
        
        except Exception as e:
            self.logger.error(f"Error in face recognition: {e}")
            return []
    
    def _apply_face_recognition(self, objects: List[Dict], faces: List[Dict]):
        """
        Apply face recognition results to detected persons.
        
        Args:
            objects: List of detected objects
            faces: List of recognized faces
        """
        # Match faces with person objects
        for obj in objects:
            if obj["type"] != "person":
                continue
            
            person_box = obj["box"]
            
            # Find the best matching face
            best_match = None
            best_iou = 0
            
            for face in faces:
                face_box = face["box"]
                iou = self._calculate_iou(person_box, face_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
            
            # If a good match is found, link the face to the person
            if best_match is not None and best_iou > 0.5:
                obj["face"] = best_match
                obj["features"] = best_match["embedding"]
    
    def _detect_actions(self, frame: np.ndarray, objects: List[Dict], tracked_objects: List) -> List[Dict]:
        """
        Detect actions in the frame based on object movements and poses.
        
        Args:
            frame: Current frame
            objects: List of detected objects
            tracked_objects: List of tracked objects
            
        Returns:
            List of detected actions
        """
        # For a full implementation, this would use an action recognition model
        # In this simplified version, we'll derive basic actions from object movements
        
        actions = []
        
        # Use tracking information to infer movements
        for track in tracked_objects:
            if not hasattr(track, 'track_id') or not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            # Find corresponding object
            obj = None
            for o in objects:
                if o.get("track_id") == track_id:
                    obj = o
                    break
            
            if obj is None:
                continue
            
            # Check if this is a person
            if obj["type"] == "person":
                # Check for movement
                if hasattr(track, 'velocity') and track.velocity is not None:
                    vel_x, vel_y = track.velocity
                    speed = (vel_x**2 + vel_y**2)**0.5
                    
                    # Determine action based on speed
                    action_type = None
                    confidence = 0.0
                    
                    if speed < 1.0:
                        action_type = "standing"
                        confidence = 0.8
                    elif speed < 5.0:
                        action_type = "walking"
                        confidence = 0.7
                    else:
                        action_type = "running"
                        confidence = 0.6
                    
                    if action_type:
                        action = {
                            "type": action_type,
                            "subject_id": obj["id"],
                            "subject_type": "person",
                            "confidence": confidence,
                            "description": f"Person {track_id} is {action_type}"
                        }
                        actions.append(action)
        
        return actions
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Compute the area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the area of union
        union_area = box1_area + box2_area - intersection_area
        
        # Compute the IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou


class SimpleTracker:
    """
    A simple object tracker for fallback when DeepSORT is not available.
    """
    
    def __init__(self):
        """Initialize the simple tracker."""
        self.tracks = []
        self.next_id = 1
    
    def update_tracks(self, detections, frame=None):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection boxes [x1, y1, x2, y2, score, class_id]
            frame: Current frame (not used in simple tracker)
            
        Returns:
            List of updated tracks
        """
        # Convert detections to standard format
        detection_boxes = np.array([[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in detections])  # [x, y, w, h]
        detection_scores = np.array([d[4] for d in detections])
        
        # If no current tracks, create new tracks for all detections
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.tracks.append(SimpleTrack(
                    self.next_id,
                    detection_boxes[i],
                    detection_scores[i]
                ))
                self.next_id += 1
            return self.tracks
        
        # Calculate IoU between all tracks and all detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det_box in enumerate(detection_boxes):
                iou_matrix[i, j] = self._calculate_iou(
                    [track.box[0], track.box[1], track.box[0] + track.box[2], track.box[1] + track.box[3]],
                    [det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3]]
                )
        
        # Assign detections to tracks using greedy algorithm
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Sort tracks by score for prioritization
        sorted_tracks = sorted(enumerate(self.tracks), key=lambda x: x[1].score, reverse=True)
        
        for track_idx, track in sorted_tracks:
            if track_idx not in unmatched_tracks:
                continue
                
            # Find the detection with highest IoU
            max_iou = 0.3  # Minimum IoU threshold
            match_det = -1
            
            for det_idx in unmatched_detections:
                iou = iou_matrix[track_idx, det_idx]
                if iou > max_iou:
                    max_iou = iou
                    match_det = det_idx
            
            if match_det >= 0:
                matched_indices.append((track_idx, match_det))
                unmatched_detections.remove(match_det)
                unmatched_tracks.remove(track_idx)
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(
                detection_boxes[det_idx],
                detection_scores[det_idx]
            )
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks.append(SimpleTrack(
                self.next_id,
                detection_boxes[det_idx],
                detection_scores[det_idx]
            ))
            self.next_id += 1
        
        # Remove stale tracks
        self.tracks = [track for track in self.tracks if track.age < 10]
        
        return self.tracks
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Compute the area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the area of union
        union_area = box1_area + box2_area - intersection_area
        
        # Compute the IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou


class SimpleTrack:
    """
    A simple track object for the SimpleTracker.
    """
    
    def __init__(self, track_id, box, score):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique identifier for this track
            box: Bounding box [x, y, width, height]
            score: Detection confidence score
        """
        self.track_id = track_id
        self.box = box.copy()
        self.score = score
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'Confirmed'  # Always confirmed for simplicity
        
        # For velocity calculation
        self.prev_box = None
        self.velocity = None
    
    def update(self, box, score):
        """
        Update the track with a new detection.
        
        Args:
            box: New bounding box
            score: New confidence score
        """
        # Save previous box for velocity calculation
        self.prev_box = self.box.copy()
        
        # Update box and score
        self.box = box.copy()
        self.score = score
        
        # Update tracklet stats
        self.hits += 1
        self.time_since_update = 0
        
        # Calculate velocity if we have previous position
        if self.prev_box is not None:
            dx = self.box[0] - self.prev_box[0]
            dy = self.box[1] - self.prev_box[1]
            self.velocity = (dx, dy)
    
    def predict(self):
        """
        Predict the next position based on velocity.
        Used when a detection is missing.
        """
        self.age += 1
        self.time_since_update += 1
        
        # Use velocity to predict next position if available
        if self.velocity is not None:
            self.box[0] += self.velocity[0]
            self.box[1] += self.velocity[1]
    
    def is_confirmed(self):
        """Check if this track is confirmed."""
        return True  # Always confirmed for simplicity
    
    def to_ltrb(self):
        """
        Convert box from [x, y, width, height] to [left, top, right, bottom].
        
        Returns:
            [left, top, right, bottom] coordinates
        """
        return [
            self.box[0],
            self.box[1],
            self.box[0] + self.box[2],
            self.box[1] + self.box[3]
        ]