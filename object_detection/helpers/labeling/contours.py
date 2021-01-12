VIDEO = "/Users/grubio/Downloads/image/camera/my_future/petra_tonica_maguary_640.mp4"
import cv2
import numpy as np

cap = cv2.VideoCapture(VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

count = 0
while cap.isOpened():
    count += 1
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    count_countour = 0
    for contour in contours:
        count_countour += 1
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        obj_width = x + y
        obj_height = y + h

        # my
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(contour)
        # box = np.int0(box)
        # cv2.drawContours(frame1, [box], 0, (0, 0, 255), 2)

        #
        rect_img = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            rect_img,
            f"w: {obj_width}/ h: {obj_height}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )
        # cv2.imwrite(f"/Users/grubio/Downloads/bounding_boxes/frame_{count_countour}_.jpg", frame1)

        # TODO make a function to generate boxes in every 5 frames
        cv2.putText(
            frame1,
            "Status: {}".format("Movement"),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    # image = cv2.resize(frame1, (1280, 720))
    # out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27 or frame2 is None:
        break
    import time
    time.sleep(0.5)

cv2.destroyAllWindows()
cap.release()
out.release()