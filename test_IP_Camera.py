import numpy as np
import cv2
from tracker import *
from datetime import datetime
import socket
import sys


tracker = EuclideanDistTracker()
cap = cv2.VideoCapture("rtsp://admin:P%40ssw%40rd%2F020@10.10.81.100/profile2/media.smp")  # Open video file
#cap = cv2.VideoCapture(r"C:\Users\met00477\Videos\Captures\video2.mp4")
count_points =[(700, 403),(700, 500),(278, 500),(278, 404),(700, 403)]

cv2.namedWindow('Fullscreen', cv2.WINDOW_KEEPRATIO)
polygon=[]
points=[]
time_ids = {}
isplaying = True 

def left_click_detect(event, x, y, flags, points):
    if (event == cv2.EVENT_LBUTTONDOWN):
        print(f"\tClick on {x}, {y}")
        points.append([x,y])
        #print(points)



def sendCounts(count):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect the socket to the port where the server is listening
    server_address = ('10.10.81.104', 6061)
    print("Connection to server")
    sock.connect(server_address)
    message = str(count)
    #print("sending "+message+" to 10.10.81.162")
    sock.sendall(message.encode())

len1 =0 
region_ids = set()
object_detector = cv2.createBackgroundSubtractorMOG2(history=0,varThreshold=100)
masked_image = None
b = False
while(cap.isOpened()):




    ret, frame = cap.read()  # read a frame
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if not ret:
        print('EOF')
        print(len(region_ids))
        break

    #roi = cv2.polylines(frame, polygon, False, (0, 0, 255), thickness=1)

    if  b:
        masked_image = cv2.bitwise_and(frame, mask)
        cv2.imshow("masked", masked_image)
        mogMask = object_detector.apply(masked_image)
        # gray_mask = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
        # gray_mask = gray_mask#[300:500,450:700]
        #cv2.imshow("gray_mask",gray_mask)
        _,threshold = cv2.threshold(mogMask,254,255,cv2.THRESH_BINARY)
        #cv2.imshow("B&W",threshold)
        contours,_ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        #count_points = [(493,0),(494,60),(537,60),(537,0),(493, 0)] #[(518,312),(468,425)]
        #count_points =[(379, 293),(629, 290),(625, 245),(381, 245),(379, 293)]
        
        #count_points = [(325,178),(325,519),(400,506),(400,178),(325,178)]
        count_region = [np.int32(count_points)]
        roi = cv2.polylines(frame, count_region, False, (0, 0, 255), thickness=2)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > 13000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
                surf =area
        len1 = len(region_ids)
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame,(x+w,y+h),3,(0,0,255),-1)
            inside_region = cv2.pointPolygonTest(np.array(count_points), (x+w,y+h),False)
            if inside_region > 0:
                region_ids.add(str(id))
            # print(region_ids)
            #now = datetime.now()
            #time_ids.update({id:now.strftime("%H:%M:%S")})

    cv2.putText(frame, "Count: "+str(len(region_ids)-1), (800, 50), cv2.FONT_HERSHEY_PLAIN, 2, (20, 0, 255), 2)
    
    cv2.imshow('Frame', frame)
    len2 = len(region_ids)
    # if(len2 > len1):
    #     sendCounts(str(len(region_ids)))
    # Abort and exit with 'Q'
    key = cv2.waitKey(1)
    if (key == ord('q')):
        break
    elif (True): 
        mask = np.ones(frame.shape,dtype=np.uint8)
        mask.fill(0)
        #points = [(532, 17),(517, 42),(607, 72),(612, 44)] 
        points = [(320, 257),(314, 479),(564, 475),(558, 271)]
        #points = [(325,178),(325,519),(716,506),(684,178)]
        polygon = [np.int32(points)]
        roi_corners = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, color=(255,255,255))
        b = True
        points = []
    
    cv2.setMouseCallback('Frame', left_click_detect, points)

for key in time_ids:
    print(str(key)+" "+str(time_ids[key]))
cap.release()  # release video  
cv2.destroyAllWindows()  # close all openCV windows



