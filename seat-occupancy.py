import torch
import cv2
import os
from PIL import Image, ExifTags
# from django.conf import settings
import numpy as np
from io import BytesIO


class Detector:
    def __init__(self, yolo_path, model_path, names_path):
        """
        Initializes the Detector class with YOLO model and configurations.
        
        Args:
            yolo_path (str): Path to the YOLO model repository
            model_path (str): Path to the YOLO model weights
            names_path (str): Path to the file containing class names
        """
        # Load YOLO model
        self.__model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
        
        # Load class names
        with open(names_path, "r") as f:
            self.__classes = [line.strip() for line in f.readlines()]

    def predict(self, image_path, iou_threshold=0.3, output_path=None):
        """
        Predicts chair occupancy in a given image.
        
        Args:
            image_path (str): Path to the input image
            iou_threshold (float, optional): IoU threshold for occupancy check
            output_path (str, optional): Path to save the annotated image
        
        Returns:
            tuple: Annotated frame and occupancy data
        """
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Unable to read image from {image_path}")

        # Detect objects
        results = self.__model(frame)
        
        # Filter detections
        chair_boxes, chair_confidences, person_boxes, person_confidences, classes_detected = self.__filter_result(results)
        
        # Apply Non-Maximum Suppression
        chair_indices = self.__perform_NMS(chair_boxes, chair_confidences, 0.5, 0.5)
        
        # Find occupied and vacant chairs
        occupied_chairs_boxes, occupied_chairs_confidences, vacant_chairs_boxes, vacant_chairs_confidences = self.__find_occupied_and_vacant_chairs(
            frame, chair_indices, chair_boxes, chair_confidences, person_boxes, iou_threshold
        )
        
        # Prepare occupancy data
        data = [
            classes_detected, 
            len(chair_boxes), 
            len(person_boxes), 
            len(occupied_chairs_boxes), 
            len(vacant_chairs_boxes)
        ]
        
        # Save annotated image if output path is provided
        if output_path:
            cv2.imwrite(output_path, frame)
        
        return frame, data

    def __filter_result(self, results):
        """
        Filter detection results based on confidence and class.
        
        Args:
            results: YOLO detection results
        
        Returns:
            Filtered boxes, confidences, and detected classes
        """
        chair_boxes, chair_confidences = [], []
        person_boxes, person_confidences = [], []
        classes_detected = []

        for det in results.xyxy[0]:
            confidence = det[4].item()
            predicted_class_id = int(det[5].item())
            class_name = self.__classes[predicted_class_id]
            
            classes_detected.append(f'{class_name} {confidence:.2f}')
            
            if confidence > 0.5 and (class_name == "chair" or class_name == "person"):
                x1, y1, x2, y2 = map(int, det[:4].tolist())
                if class_name == "chair":
                    chair_boxes.append([x1, y1, x2, y2])
                    chair_confidences.append(float(confidence))
                elif class_name == "person":
                    person_boxes.append([x1, y1, x2, y2])
                    person_confidences.append(float(confidence))
        return chair_boxes, chair_confidences, person_boxes, person_confidences, classes_detected

    def __perform_NMS(self, boxes, confidences, score, nms):
        """
        Perform Non-Maximum Suppression on detected boxes.
        
        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            score: Confidence threshold
            nms: IoU threshold for NMS
        
        Returns:
            Indices of boxes after NMS
        """
        return cv2.dnn.NMSBoxes(boxes, confidences, score, nms)

    def __find_occupied_and_vacant_chairs(self, frame, chair_indices, chair_boxes, chair_confidences, person_boxes, iou_threshold):
        """
        Determine occupied and vacant chairs.
        
        Args:
            frame: Input image
            chair_indices: Indices of chairs after NMS
            chair_boxes: Detected chair bounding boxes
            chair_confidences: Chair detection confidences
            person_boxes: Detected person bounding boxes
            iou_threshold: IoU threshold for occupancy
        
        Returns:
            Occupied and vacant chair information
        """
        occupied_chairs_boxes, occupied_chairs_confidences = [], []
        vacant_chairs_boxes, vacant_chairs_confidences = [], []

        for i in chair_indices:
            chair_box = chair_boxes[i]
            chair_confidence = chair_confidences[i]
            
            is_vacant = self.__check_vacant_chair(chair_box, person_boxes, iou_threshold)
            
            if is_vacant:
                vacant_chairs_boxes.append(chair_box)
                vacant_chairs_confidences.append(chair_confidence)
                # self.__draw_bounding_boxes(frame, chair_box, chair_confidence, "Vacant", (50, 205, 50))
            else:
                occupied_chairs_boxes.append(chair_box)
                occupied_chairs_confidences.append(chair_confidence)
                # self.__draw_bounding_boxes(frame, chair_box, chair_confidence, "Occupied", (250, 128, 114))
        i = vacant_chairs_confidences.index(max(vacant_chairs_confidences))
        self.__draw_bounding_boxes(frame, vacant_chairs_boxes[i], vacant_chairs_confidences[i], "Most Vacant", (50, 205, 50))
        return occupied_chairs_boxes, occupied_chairs_confidences, vacant_chairs_boxes, vacant_chairs_confidences

    def __check_vacant_chair(self, chair_box, person_boxes, iou_threshold):
        """
        Check if a chair is vacant based on IoU with person boxes.
        
        Args:
            chair_box: Chair bounding box
            person_boxes: List of person bounding boxes
            iou_threshold: IoU threshold for occupancy
        
        Returns:
            Boolean indicating chair vacancy
        """
        for person_box in person_boxes:
            iou = self.__calculate_iou(person_box, chair_box)
            if iou > iou_threshold:
                return False
        return True

    def __calculate_iou(self, person_box, chair_box):
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            person_box: Person bounding box
            chair_box: Chair bounding box
        
        Returns:
            IoU score
        """
        intersection_x1 = max(person_box[0], chair_box[0])
        intersection_y1 = max(person_box[1], chair_box[1])
        intersection_x2 = min(person_box[2], chair_box[2])
        intersection_y2 = min(person_box[3], chair_box[3])
        
        intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
        
        union_area = (
            (person_box[2] - person_box[0] + 1) * (person_box[3] - person_box[1] + 1) +
            (chair_box[2] - chair_box[0] + 1) * (chair_box[3] - chair_box[1] + 1) -
            intersection_area
        )
        
        return intersection_area / union_area

    def __draw_bounding_boxes(self, frame, box, confidence, cls, color):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input image
            box: Bounding box coordinates
            confidence: Detection confidence
            cls: Class label
            color: Bounding box color
        """
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f'{cls} {confidence:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        
        cv2.rectangle(frame, (x1, y1), (x1 + max(text_size[0], x2 - x1), y1 + 20), color, -1)
        cv2.putText(frame, text, (x1, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

def image_converter(original_image, dimension):
    # Get the EXIF data from the image
    try:
        exif_data = original_image._getexif()
    except AttributeError:
        exif_data = None

    # Rotate the image according to the EXIF orientation data
    if exif_data is not None:
        for tag, value in exif_data.items():
            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                if value == 3:
                    original_image = original_image.rotate(180, expand=True)
                elif value == 6:
                    original_image = original_image.rotate(270, expand=True)
                elif value == 8:
                    original_image = original_image.rotate(90, expand=True)
                break
            
    # Resize the image while maintaining aspect ratio
    original_image.thumbnail(dimension)

    # Create a new blank image with the target size
    converted_image = Image.new("RGB", dimension)

    # Paste the resized image onto the new image
    x_offset = (dimension[0] - original_image.size[0]) // 2
    y_offset = (dimension[1] - original_image.size[1]) // 2
    converted_image.paste(original_image, (x_offset, y_offset))

    return converted_image

def main():
    # Example usage
    yolo_path = "/Users/kaushalpatil/Development/HackSC /utils/yolov3"
    model_path = "/Users/kaushalpatil/Development/HackSC /utils/yolov3/yolov3.pt"
    names_path = "/Users/kaushalpatil/Development/HackSC /utils/coco.names"
    # Instantiate the model processing class
    detector = Detector(yolo_path, model_path, names_path)

    detector = Detector(yolo_path, model_path, names_path)
    
    try:
        # PILimage = Image.open("/Users/kaushalpatil/Development/HackSC /classroom.jpeg")
        # width, height = PILimage.size
        # print(width,height)
        
        # # Convert the image to 640*640
        # if (width>640 or height >640):
        #     PILimage = image_converter(PILimage, (640, 640))

        # # image_bytes = frame.read()
        # # Convert the image to bytes
        # buffer = BytesIO()
        # PILimage.save(buffer, format="JPEG")
        # image_bytes = buffer.getvalue()
        # image_np = np.frombuffer(image_bytes, np.uint8)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        frame, data = detector.predict(input("Enter the file string: "))
        
        print("Detected Classes:", data[0])
        print("Total Chairs:", data[1])
        print("Total Persons:", data[2])
        print("Occupied Chairs:", data[3])
        print("Vacant Chairs:", data[4])
        
        # Optional: Display the annotated image
        # cv2.imshow('Seat Occupancy', frame)
        try:
            output_path = os.path.join("/Users/kaushalpatil/Development/HackSC /results", 'frame.jpg')
            success = cv2.imwrite(output_path, frame)
            if not success:
                print("Failed to save image")
        except Exception as e:
            print(f"Error saving frame: {e}")
        exit()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print("hello")
    
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()