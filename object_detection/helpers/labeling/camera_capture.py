import cv2
import time
from labels import generate_file_labels
import os

cap = cv2.VideoCapture(0)

CAPTURE_INTERVAL = 15.0
frame_count = 0
frame_rate = 10
prev = 0

label_name = "brahma_puro_malte_350ml"
file_path = f"/Users/grubio/Downloads/bounding_boxes2/{label_name}/"
count_saved_images = 0

os.makedirs(file_path, exist_ok=True)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
record = True

# todo remove hardcode
# x = 800
# y = 50
# w = 300  # BLACK PRINCESS VERTICAL
# h = 600

# x = 800
# y = 300
# w = 300 #TODAS CERVEJAS DE LATA (TAMPA)
# h = 300

# x = 800
# y = 300
# w = 300 #CERVEJA NA VERTICAL
# h = 300

# x = 800
# y = 150
# w = 300  # BLACK PRINCESS VERTICAL
# h = 550

# x = 830
# y = 350
# w = 220  # Cerveja long neck (tripe elevado no 3 a esquerda)
# h = 700

x = 830
y = 500
w = 220  # Patagonia 475ml  (tripe elevado no 3 a esquerda)
h = 480

draw_box= False


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    if draw_box:
        rect_img = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            rect_img,
            f"{label_name}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    cv2.imshow("frame", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    # only capture frame every one second
    time_elapsed = time.time() - prev

    if time_elapsed > CAPTURE_INTERVAL / frame_rate:
        count_saved_images += 1
        prev = time.time()
        file_name = f"{label_name}_{frame_count}_{time.time()}"

        crop_img = frame1[y: y + h, x: x + w]
        if record:
            if draw_box:
                cv2.imwrite(f"{file_path}/{file_name}.jpg", crop_img)
            else:
                cv2.imwrite(f"{file_path}/{file_name}.jpg", frame)
            # generate_file_labels(
            #     label_name, f"{file_name}, file_path, w, h, 0, 0, w, h
            # )
            print(f"scount_saved_images {count_saved_images}")

    # c = cv2.waitKey(0) % 256

    # # esc key
    # if c == ord('s'):
    #     cv2.destroyAllWindows()
    #     break
    # elif c == ord('r'):
    #     if record:
    #         record = False
    #         print("Pause recording")
    #     else:
    #         record = True
    if cv2.waitKey(40) == 27 or frame2 is None:
        break
    if count_saved_images == 300:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
