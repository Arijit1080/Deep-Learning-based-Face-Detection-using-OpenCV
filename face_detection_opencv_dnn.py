import cv2
import time
import sys
import os
from imutils.video import VideoStream
import imutils

def detectFace(net, frame):
    frame_copy = frame.copy()
    frameHeight = frame_copy.shape[0]
    frameWidth = frame_copy.shape[1]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frame_copy, bboxes

if __name__ == "__main__" :

    DNN = "TF"  #According to which model you want to use, change this

    if DNN == "CAFFE":      #FP16 version of the original caffe implementation
        modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "model/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:                   #8 bit Quantized version using Tensorflow
        modelFile = 'model/opencv_face_detector_uint8.pb'
        configFile = 'model/opencv_face_detector.pbtxt'
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    print("->Starting Face Detection")
    c = VideoStream(src=0).start()
    time.sleep(2.0)

    count_image=1  #Counts number of frames/images captured.

    while True:

        frame = c.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        outOpencvDnn, bboxes = detectFace(net,frame)

        cv2.imshow("Face Detection", outOpencvDnn)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key % 256 == 32 :  # If spacebar is pressed.
            i_name="OpenCV_frame{}.png".format(count_image)
            cv2.imwrite(i_name,outOpencvDnn)
            image_count +=1

    cv2.destroyAllWindows()
    c.stop()