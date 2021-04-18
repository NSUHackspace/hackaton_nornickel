import numpy as np
import cv2
import statistics
import csv
import sys
import morph

"""
    Authors: nsu-hackspace-team
    
    Hydra provides instrumentation for image-based froth analysis
    based on features dynamics. Detected features are related 
    to froth movement. Features flow characterized by median velocity
    and direction. Dispersion of direction provides ability to 
    detect waves on froth surface.

"""


class Hydra:
       
    def __init__(self, tracked_elements = 100):
        self.tracked_elements = tracked_elements        
        self.color = np.random.randint(0, 255, (tracked_elements, 3))
        # parameters for tracking statistics
        self.normal_flow_variance = 0.7

        self.memory_depth = 10 
        self.wave_memory = [0] * self.memory_depth 
   
        self.isWave = False
        # parameters for Shi-Tomasi corner detector
        self.feature_params = dict( maxCorners = tracked_elements,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)
        # parameters for lucas kanade optical flow routine
        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



    def init_features(self, base_frame):
        self.old_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)      
        self.mask = np.zeros_like(base_frame)


    def calc_vec_direction(x, y):
        vect = [x + 0.001, y + 0.001]
        unit_v = vect / np.linalg.norm(vect)
        dot_product = np.dot(unit_v, [1, 0])
        angle = np.arccos(dot_product)
        return angle
   

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        # draw the tracks
        angles = []
        velocities = []

        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel() 
            c,d = old.ravel()
            angles.append(Hydra.calc_vec_direction(c-a, d-b))
            velocities.append(np.linalg.norm([c-a, d-b]))

            self.mask = cv2.line(self.mask, (a,b), (c,d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, self.color[i].tolist(), -1)   

        img = cv2.add(frame, self.mask)
    
        if (len(angles) > 2):
            del self.wave_memory[0]        
            self.wave_memory.append(statistics.variance(angles))
            self.direction = statistics.median(angles) * 180 / np.pi + 180
            self.velocity = statistics.median(velocities)
            self.isWave = sum([1 for x in self.wave_memory if x > self.normal_flow_variance]) > (self.memory_depth / 3)

        
        self.print_metrics(img)


        self.old_gray = frame_gray.copy()
    
        if (len(good_new) >= self.tracked_elements * 0.5) :
            self.p0 = good_new.reshape(-1,1,2)
        else:
            self.init_features(frame) 
        return img

    def print_metrics(self, img):
        
        font_name = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_clr = (255, 0, 0)
        thickness = 2
        
        font_org = (50, 50)
        img = cv2.putText(img, 'speed: ' + str(round(self.velocity, 2)), font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)
        
        font_org = (50, 80)
        img = cv2.putText(img, 'dir: ' + str(round(self.direction, 2)), font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)
        
        font_org = (50, 110)
        if self.isWave:
            img = cv2.putText(img, 'wave', font_org, font_name,
                   font_scale, font_clr, thickness, cv2.LINE_AA)
        return img

    # Metrics collection: median flow velocity, median flow direction, dispersion, wave detection result
    def log_str(self):
        return [round(self.velocity, 3), 
                round(self.direction, 3),
                round(self.wave_memory[-1], 3), self.isWave] 



def main():
    window_name_h = 'Hydra'
    window_name_m = 'Morph'

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


    h = Hydra(100)
    m = morph.Morph(fps)

    # Optical flow processing 
    ret, frame = cap.read()
    frame = frame[0:y_end, x_start:x_end]  

    h.init_features(frame)
    
    metrics_file = filename + '.csv'
    with open(metrics_file, mode='w') as metrics:
            metrics_writer = csv.writer(metrics, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            metrics_writer.writerow(['median velocity', 'median direction', 'direction dispersion', 'is wave', '#contours', 'mass center', 'is steady'])
    
    while (True):
        ret, frame = cap.read()
        if np.shape(frame) == ():
            break

        frame1 = frame[0:y_end, x_start:x_end]
        frame2 = frame1.copy()
        frame1 = h.process_frame(frame1)
        frame2 = m.process_frame(frame2) 
        
        img_stack = np.hstack((frame1, frame2))
        #cv2.imshow(window_name_h, frame1)
        #cv2.imshow(window_name_m, frame2)
        cv2.imshow("Hydra & Morph", img_stack)
        k = cv2.waitKey(20) & 0xff
        
        with open(metrics_file, mode='a') as metrics:
            metrics_writer = csv.writer(metrics, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            metrics_writer.writerow(h.log_str() + m.log_str())
        if k == 27:
            break


        
        
    cv2.destroyAllWindows()
    cap.release()

### RUN CORE APPLICATION
if __name__ == '__main__':
  main()
