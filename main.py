import mediapipe as mp
import numpy as np
from typing import Tuple, Union, Any
import cv2
import time 
import math
import asyncio


def is_valid_normalized_value(value: float) -> bool:
    """Check if value is a normalized value."""
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# def color_classifier(image):
#     """Classify image as green, black, or neither based on color."""
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     lower_green = np.array([36, 25, 25])
#     upper_green = np.array([90, 255, 255])
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 30])

#     mask_green = cv2.inRange(hsv, lower_green, upper_green)
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)

#     count_green = cv2.countNonZero(mask_green)
#     count_black = cv2.countNonZero(mask_black)

#     max_count = max(count_green, count_black)
    
#     if max_count == count_green:
#         return "green"
#     elif max_count == count_black:
#         return "black"
#     else:
#         return "neither"
    
def color_classifier(image):
    """Classify image as green, black, blue, red, white, or neither based on color."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])
    
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([90, 25, 25])
    upper_blue = np.array([150, 255, 255])
    
    lower_red = np.array([0, 25, 25])
    upper_red = np.array([10, 255, 255])
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    count_green = cv2.countNonZero(mask_green)
    count_black = cv2.countNonZero(mask_black)
    count_blue = cv2.countNonZero(mask_blue)
    count_red = cv2.countNonZero(mask_red)
    count_white = cv2.countNonZero(mask_white)

    max_count = max(count_green, count_black, count_blue, count_red, count_white)
    
    if max_count == count_green:
        return "green"
    elif max_count == count_black:
        return "black"
    # elif max_count == count_blue:
    #     return "blue"
    # elif max_count == count_red:
    #     return "red"
    # elif max_count == count_white:
    #     return "white"


def draw_image(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and returns it."""
    detected_color = "None"  # Default color when no color is detected
    if detection_result is None:
        return image, detected_color
    
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        new_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        
        detected_color = color_classifier(new_image)
        
        if detected_color == "green":
            color_code = (0, 255, 0)  # Green color
        elif detected_color == "black":
            color_code = (0, 0, 0)  # Black color
        else:
            continue

        cv2.rectangle(image, start_point, end_point, color_code, 3)
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (20 + bbox.origin_x, 20 + 10 + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, color_code, 1)

    return image, detected_color

# object detection using mediapipe

model_path = 'weights/efficientdet_lite0.tflite'

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

objectDetectionResult = None

def objectDetectionCallback(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global objectDetectionResult
    objectDetectionResult = result

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=1,
    result_callback=objectDetectionCallback,
    score_threshold=0.85,
    category_allowlist=["person"]
)

# resize_width = 640
# resize_height = 480


async def play_video(video_path, window_name):
    video_capture = cv2.VideoCapture(video_path)
    resize_width = 1366
    resize_height = 768
    while True:
        ret, frame = video_capture.read()
        if ret:
            # Resize the frame
            frame = cv2.resize(frame, (resize_width, resize_height))
            
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
        await asyncio.sleep(0.03)  # Delay of 30 milliseconds (approximately 33 frames per second)

    video_capture.release()
    cv2.destroyAllWindows()


async def main():
    with ObjectDetector.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)

        # Load videos
        # video_a = cv2.VideoCapture('HBO-intro.mp4')
        # video_b = cv2.VideoCapture('HBO-closing.mp4')
        video_a = 'HBO-intro.mp4'
        video_b = 'HBO-closing.mp4'

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = mp.Timestamp.from_seconds(time.time()).microseconds()
            detector.detect_async(mp_image, frame_timestamp_ms)

            drawn_image, color = draw_image(mp_image.numpy_view(), objectDetectionResult)
            print(color)

            if color == "green":
                # Run video A coroutine
                await asyncio.sleep(1)
                await play_video(video_a, 'green video')
                await asyncio.sleep(3)  # Add a delay of 1 seconds after video A is closed

            elif color == "black":
                # Run video B coroutine
                await asyncio.sleep(1)
                await play_video(video_b, 'black video')
                await asyncio.sleep(3)  # Add a delay of 1 seconds after video B is closed

            cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Object Detection', drawn_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0)

        cap.release()
        cv2.destroyAllWindows()

# Run the main coroutine
asyncio.run(main())
