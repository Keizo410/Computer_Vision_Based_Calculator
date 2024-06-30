import cv2
from utils.caluclation import Calculator
from utils.recognition import HandwritingRecognizer
from utils.hand_detection import HandDetector
from utils.drawing import DrawingManager
import numpy as np 
import time 

def start():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    drawing_manager = DrawingManager()
    recognizer = HandwritingRecognizer()
    calculator = Calculator()
    canvas = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype='uint8')
    canvas.fill(255) 

    # Get dimensions of the video frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter objects for canvas and image
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_canvas = cv2.VideoWriter('output_canvas_com.mp4', fourcc, 20.0, (frame_width, frame_height))
    out_image = cv2.VideoWriter('output_image_com.mp4', fourcc, 20.0, (frame_width, frame_height))


    while cap.isOpened():
        success, image = cap.read()
        predictions = ""
        result = ""
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        landmarks, image = hand_detector.detect_hands(image)

        if landmarks:

            canvas, image = drawing_manager.update(landmarks, image)

            if drawing_manager.get_isComplete():
                eq = canvas.copy()
                predictions = recognizer.visualize_predictions(eq)
                print(predictions)
                result = calculator.calculate(predictions)
                print(result)
                drawing_manager.display_result(canvas, result)
                # Display the result for 5 seconds
                start_time = time.time()
                while time.time() - start_time < 1:
                    # cv2.imshow("answer", canvas)
                    calculator.reset()
                    if cv2.waitKey(1) & 0xFF == 27:
                        break


        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow("Drawn Lines", canvas)

        # Write frames into the respective video files
        if canvas.shape[-1] != 3:  # Check if canvas has 3 channels (BGR)
            vid_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)  # Convert to BGR if necessary
        out_canvas.write(vid_canvas)
        out_image.write(image)
        
        # Exit the loop when 'ESC' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release everything when finished
    cap.release()
    out_canvas.release()
    out_image.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()
