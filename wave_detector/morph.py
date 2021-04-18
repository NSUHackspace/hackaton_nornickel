import cv2
import numpy as np
import statistics
import sys

"""
    Authors: nsu-hackspace-team
    
    Morph provides instrumentation for image-based froth analysis
    based on contour structure. Detected countours are related 
    to froth glares. Countours system characterized by speed 
    of change (related to steadiness of froth flow) and integral
    parameter of detected bubbles mass center.

"""


class Morph:
    def __init__(self, fps):
        self.memory_depth = int(fps)
        self.contours_memory = [0] * self.memory_depth 
        self.isSteady = False

    
    def bubbles_mass_center(bubble_centers, width, height):
        x_mass = 0
        y_mass = 0
        for (x, y) in bubble_centers:
            x_mass += x
            y_mass += y

        x_mass = round(x_mass * 100 / (len(bubble_centers) * width), 2)
        y_mass = round(y_mass * 100 / (len(bubble_centers) * height), 2)
        return (x_mass, y_mass) 


    def auto_canny(image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged

    # Calculate speed of morphology change
    def countours_slope_change(self):
        mean = statistics.mean(self.contours_memory)
        diff1 = (mean - self.contours_memory[0]) / (self.memory_depth / 2)
        diff2 = (self.contours_memory[-1] - mean) / (self.memory_depth / 2)
        self.isSteady = (abs(diff1) < 2 and abs(diff2) < 2)


    def process_frame(self, frame):
        h, w, d = frame.shape
    
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv_frame)

        core = 5
        bilateral = cv2.bilateralFilter(V, core, 25, 25)
        contours, hierarchy = cv2.findContours(Morph.auto_canny(bilateral), 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center_points = []
    
        canvas_cb = np.zeros((h, w), dtype=np.uint8)
        
        for c in contours:      
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / (M["m00"] + 1e-5))
                cY = int(M["m01"] / (M["m00"] + 1e-5))
                center_points.append((cX, cY))
                (x,y), r = cv2.minEnclosingCircle(c)
                radius = int(r)
                cv2.circle(frame, (cX, cY), radius, (0, 255, 0), 1)

        del self.contours_memory[0]
        self.contours_memory.append(len(contours))
        self.mass_center = Morph.bubbles_mass_center(center_points, w, h)
        self.countours_slope_change()
        self.print_metrics(frame)
        return frame


    def print_metrics(self, frame):
        font_name = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_clr = (255, 0, 0)
        thickness = 2
        
        font_org = (50, 50)
        img = cv2.putText(frame, 'cont: ' + str(self.contours_memory[-1]), font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)
        
        font_org = (50, 80)
        img = cv2.putText(frame, 'mass: ' + str(self.mass_center), font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)

        font_org = (50, 110)
        if not self.isSteady:
            img = cv2.putText(img, 'unsteady', font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)
        return frame

    # Metrics collection: number of contours, bubbles mass center, froth flow steadiness 
    def log_str(self):
        return [self.contours_memory[-1],
                self.mass_center,
                self.isSteady]



def main():
    window_name = 'Morph'
    if (len(sys.argv) < 2):
        print('type video filename')
        exit()
    filename = sys.argv[1]
    cap = cv2.VideoCapture(filename) 
    fps = cap.get(cv2.CAP_PROP_FPS)    

    # Calculate ROI
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    x_start = int(frame_width / 8) 
    x_end = int(frame_width)
    y_end = int(frame_height * 4 / 5)
    m = Morph(fps)

    while(True):
        ret, frame = cap.read()
        if frame is None:
            break
        
        frame = frame[0:y_end, x_start:x_end]
        m.process_frame(frame)         
        #print(m.log_str())
        cv2.imshow(window_name, frame)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


### RUN CORE APPLICATION
if __name__ == '__main__':
  main()

