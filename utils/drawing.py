import numpy
import cv2 
import time 
import numpy as np 

class DrawingManager:
    def __init__(self):
        self.drawer = []
        self.segments = []
        self.eraser = []
        self.drawer_cont = 0
        self.isErasing = False
        self.isDrawing = False
        self.isReset = False
        self.isComplete = False
        self.movement_detected = False
        self.drawing_start_time = None
        self.drawing_delay = 0.7  # Delay in seconds before starting to store points
        self.complete_start_time= None

    def update(self, hands, image):
        # update the drawing and erasing info.
        # if user's index finger is up, then drawing starts. Erasing stops.
        # if user's index and middle finger is up, then erasing starts. Drawing stops.
        # drawing cordinate
        canvas = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        canvas.fill(255) 
        try:
            # h, w, _ = image.shape
            for hand in hands:
                
                # Get landmarks coordinates
                landmarks = hand.landmark
                positions = self.get_landmark_positions(landmarks, image)

                # Constantly checks if it is drawing or reseting or..(finger up and down)
                self.set_erasing(landmarks)
                self.set_drawing(landmarks)
                self.set_reset(landmarks)
                self.set_complete(landmarks)
                
                index_tip_pos = positions['index_tip']
                middle_tip_pos = positions['middle_tip']
            
                draw_point = self.get_average_point(index_tip_pos, middle_tip_pos)
                # erase_point = self.get_average_point(index_tip_pos, middle_tip_pos, ring_tip_pos)

                if self.isDrawing:
                    if self.drawing_start_time is None:
                        self.drawing_start_time = time.time()
                    elif time.time() - self.drawing_start_time >= self.drawing_delay:
                        self.drawer.append(draw_point)
                    
                    print("drawing!")
                elif self.isReset:
                    print("reset!")
                    self.reset_drawing()
                elif self.isComplete:
                    print("complete!")
                else:
                    self.stop_drawing()
                
                self.render_drawing(canvas)
                self.render_drawing(image)

        except Exception as e:
            print(e)
        
        return canvas, image


    def stop_drawing(self):
        self.segments.append(self.drawer)
        self.drawer = []
        self.drawing_start_time = None

    def get_average_point(self, *points):
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        n = len(points)
        return (int(x_sum / n), int(y_sum / n))

    def set_erasing(self, landmarks):
        index = self.is_index_up(landmarks)
        middle = self.is_middle_up(landmarks)
        ring = self.is_ring_up(landmarks)
        self.isErasing = index and middle and ring

    def set_drawing(self, landmarks):
        thumb = self.is_thumb_up(landmarks)
        index = self.is_index_up(landmarks)
        middle = self.is_middle_up(landmarks)
        ring = self.is_ring_up(landmarks)
        pinky = self.is_pinky_up(landmarks)
        self.isDrawing = index and middle and not ring and not pinky and thumb
    
    def set_reset(self, landmarks):
        thumb = self.is_thumb_up(landmarks)
        index = self.is_index_up(landmarks)
        middle = self.is_middle_up(landmarks)
        ring = self.is_ring_up(landmarks)
        pinky = self.is_pinky_up(landmarks)
        self.isReset = not index and not middle and not ring and not pinky and thumb

    def set_complete(self, landmarks):
        thumb = self.is_thumb_up(landmarks)
        index = self.is_index_up(landmarks)
        middle = self.is_middle_up(landmarks)
        ring = self.is_ring_up(landmarks)
        pinky = self.is_pinky_up(landmarks)
        # self.isComplete = index and middle and ring and thumb and pinky
        if thumb and index and middle and ring and pinky:
            if self.complete_start_time is None:
                self.complete_start_time = time.time()
        else:
            self.complete_start_time = None
        self.isComplete = self.complete_start_time is not None and time.time() - self.complete_start_time >= 3  # 3 seconds for complete gesture


    def get_isComplete(self):
        return self.isComplete
    
    def is_thumb_up(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_pj = landmarks[3]
        return thumb_tip.y < thumb_pj.y

    def is_index_up(self, landmarks):
        # return True if previous joint y coordinate is below the tips coordinate
        index_tip = landmarks[8]
        index_pj = landmarks[7]
        return index_tip.y < index_pj.y

    def is_middle_up(self, landmarks):
        # return True if previous joint y coordinate is below the tips coordinate
        middle_tip = landmarks[12]
        middle_pj = landmarks[11]
        return middle_tip.y < middle_pj.y
    
    def is_ring_up(self, landmarks):
        ring_tip = landmarks[16]
        ring_pj = landmarks[15]
        return ring_tip.y < ring_pj.y
    
    def is_pinky_up(self, landmarks):
        pinky_tip = landmarks[20]
        pinky_pj = landmarks[19]
        return pinky_tip.y < pinky_pj.y

    def get_landmark_positions(self, landmarks, image):
        h, w, _ = image.shape
        positions = {
            'thumb_tip': (int(landmarks[4].x * w), int(landmarks[4].y * h)),
            'index_tip': (int(landmarks[8].x * w), int(landmarks[8].y * h)),
            'middle_tip': (int(landmarks[12].x * w), int(landmarks[12].y * h)),
            'ring_tip': (int(landmarks[16].x * w), int(landmarks[16].y * h)),
            'pinky_tip': (int(landmarks[20].x * w), int(landmarks[20].y * h))
        }
        return positions

    def is_moving_activity(self, point1, point2, threshold=2):
        # Calculate the Euclidean distance between two points
        distance = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
        # If the distance exceeds the threshold, consider it as drawing activity
        return distance > threshold

    def get_drawing(self):
        return self.drawer
    
    def get_movement_detected(self):
        return self.movement_detected
    
    def is_erasing(self):
        return self.isErasing

    def is_drawing(self):
        return self.isDrawing

    def render_drawing(self, image):
        for segment in self.segments:
            for i in range(1, len(segment)):
                cv2.line(image, segment[i - 1], segment[i], (0, 255, 0), 5)
        
        for i in range(1, len(self.drawer)):
            cv2.line(image, self.drawer[i - 1], self.drawer[i], (0, 255, 0), 5)
    
    def count_segments(self):
        return len([segment for segment in self.segments if segment])
    
    def erase_lines(self, image):
        for eraser_point in self.eraser:
            for i in range(len(self.segments)):
                segment = self.segments[i]
                j = 0
                while j < len(segment) - 1:
                    start_point = segment[j]
                    end_point = segment[j + 1]
                    if self.line_intersects_with_point(start_point, end_point, eraser_point):
                        # Remove the line segment from the segment list
                        self.segments[i] = segment[:j] + segment[j + 1:]
                    else:
                        j += 1

    def line_intersects_with_point(self, start_point, end_point, erase_point):
        # Convert points to numpy arrays for easier calculations
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        erase_point = np.array(erase_point)

        # Calculate the distance from the erase point to the line segment
        distance = np.linalg.norm(np.cross(end_point - start_point, start_point - erase_point)) / np.linalg.norm(end_point - start_point)

        # If the distance is small enough, the erase point is close to the line
        return distance < 5  # Adjust this threshold as needed

    def reset_drawing(self):
        self.segments = []
        self.drawer = []

    def display_result(self, image, number):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 50)
        fontScale = 2
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(image, str(number), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


        
