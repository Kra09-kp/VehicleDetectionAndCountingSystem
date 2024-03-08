import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model=YOLO('yolov8s.pt') #which is pretrained model


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('vidyolov8.mp4')
# now i want to save my detections in a video
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output3.avi', fourcc, 20.0, (1020, 500))


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
cy1 = 200
cy2 = 229
cx1 = 0
cx2 = 0
offset = 6  # for the line thickness (can customize according to the video)
#area = [(245,233),(250,256),(583,220),(552,207)] # for other videos we just need to update the area coordinates
CounterU = set()
CounterD = set()
#Counter = set()
vh_down = {} # to store the vehicles which are going down
vh_up = {} # to store the vehicles which are going up
tracker = Tracker()
area_c = set()
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    #print(results)
    a = results[0].boxes.xyxy
    c = results[0].boxes.cls
    conf = results[0].boxes.conf
    if a.numel() == 0 or c.numel() == 0 or conf.numel() == 0:
        print("No detections in this frame, skipping...")
        continue
    pc = pd.DataFrame(c).astype("float")
    pconf = pd.DataFrame(conf).astype("float")
    # now add pc and pconf to a px dataframe
    #print(pc)
    #print(pconf)

    
        
    px = pd.DataFrame(a).astype("float")
    px = pd.concat([px, pconf, pc], axis=1)

    # to show full dataframe with all the columns and rows
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    # to change the columns names to synchronize with the original dataframe
    
    # to check if any of the column in dataframe is empty or not, if it is empty then we will continue to the next frame
    #print(px)
    if px.empty:
        print("No detections in this frame, skipping...")
        continue
    px.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']
    #print(px)
    #print(c)
    #print(conf)
    l = [] # to store the coordinates of the vehicles
    for index,row in px.iterrows():
        #print(row)
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        clas = class_list[int(row['class'])]
        
        #if there is a vehicle then only we will track it for this we need to check in coco.txt and get all the vehicle type and then check them
        if str(clas) in {'car','truck','bus','bicycle','motorcycle'}:
            c = clas
            l.append([x1,y1,x2,y2])
    bbox_id = tracker.update(l)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox 
        # to draw a center point on our frame
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        print(cy,id)
        # to check if the center of the vehicle is in the area or not
        #result = cv2.pointPolygonTest(np.array(area,np.int32), (cx, cy), False) 
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                # to draw a rectangle on our frame
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)
                # to put the class name on the rectangle in our frame
        cv2.putText(frame, str(c),(x3+2, y3+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)
        
        #for going down side
        if cy1<(cy+offset) and cy1>(cy-offset):
            vh_down[id]=cy
        if id in vh_down:
            if cy2<(cy+offset) and cy2>(cy-offset):
                CounterD.add(id)
        #if result >=0:
        '''cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                # to draw a rectangle on our frame
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)
                # to put the class name on the rectangle in our frame
        cv2.putText(frame, str(c)+' '+str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
        #area_c.add(id)'''
                
        #Counter.add(id)
        
        # for going up side
        if cy2<(cy+offset) and cy2>=(cy-offset):
            vh_up[id]=cy
        if id in vh_up:
            if cy1<(cy+offset) and cy1>(cy-offset):
                CounterU.add(id)
                
                ''' cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                # to draw a rectangle on our frame
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)
                # to put the class name on the rectangle in our frame
                cv2.putText(frame, str(c), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)'''
                
        # to draw a polyline for detection area
        #cv2.polylines(frame, [np.array(area,np.int32)], True, (0, 255, 0), 2)
    #cv2.putText(frame, "Total Vehicles: " + str(len(Counter)), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Going Down: " + str(len(CounterD)), (850, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, "Going Up: " + str(len(CounterU)), (850, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
    # to draw a line
    
    cv2.line(frame, (225,cy1), (556,cy1), (255,255,255),1)
    # to put text on the line
    cv2.putText(frame, "Lane 1", (220,cy1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    
    cv2.line(frame, (243,cy2), (590,cy2), (255,255,255),1)
   
    cv2.putText(frame, "Lane 2", (239,cy2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    #print(Counter)
    print(CounterD)
    print(CounterU)
    output_video.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
