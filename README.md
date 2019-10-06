# Deep Learning based Face Detection using openCV DNN

Simple code in python using openCV (DNN) for Face Detection.

## Dependencies

1. Python 3
2. opencv (above 3.3) #Watch https://youtu.be/xlmJsTeZL3w this video for opencv installation in Raspberry Pi
3. imutils (tested with 0.5.3) #sudo pip install imutils

## Run 

```
Python3 face_detect_opencv_dnn.py
```
* Don't forget to download the 'model' folder

## Setups for PC

To use default webcam
```
c = VideoStream(src=0).start()

```

To use external webcam
```
c = VideoStream(src=1).start() #If system has more than one webcam use the 'src' number accordingly

```

## Authors

**Arijit Das** 


## Acknowledgments

* https://becominghuman.ai/



