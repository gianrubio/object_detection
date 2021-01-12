import cv2, time
frame_rate = 10
prev = 0


def get_fps(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


 
video = cv2.VideoCapture(0) #"rtsp://admin:RNXTDK@192.168.0.19:554/onvif1")


get_fps(video)

while(True):
    time_elapsed = time.time() - prev
    ret, frame = video.read()

    if ret:
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            frame = cv2.resize(frame, (640, 480))

            cv2.imshow('frame',frame)
    else:
        raise Exception("no video")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()



