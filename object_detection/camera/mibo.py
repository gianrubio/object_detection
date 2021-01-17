import cv2
import time
frame_rate = 10
prev = 0


def get_fps(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


# video = cv2.VideoCapture("/Users/grubio/Downloads/test2.mp4")
video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
out = cv2.VideoWriter('/Users/grubio/Downloads/output.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, size)

get_fps(video)
i = 0
while(video.isOpened()):
    i += 1
    time_elapsed = time.time() - prev
    ret, frame = video.read()

    if ret:
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            # frame = cv2.resize(frame, (640, 480))
            # cv2.imwrite('/Users/grubio/Downloads/kang'+str(i)+'.jpg',frame)
            cv2.imshow("iphone", frame)

    else:
        break
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
video.release()
cv2.destroyAllWindows()
