#Trainning dataset every picture as the same as position
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import serial, time
from datetime import datetime
from serial_control import light_control as lc
import os

import matplotlib.pyplot as plt
import cvlib as _cvlib
from cvlib.object_detection import draw_bbox
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--frame", required=True,help="path to the image file")
args = vars(ap.parse_args())
"""
img = cv2.VideoCapture(0)
sizeTarget = (224, 224)
np.set_printoptions(suppress=True)
dataObj = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)



model = load_model("D:/Sample picture 24-June-21/4_Machine Deep Learning REV-D/keras_model.h5")  # path model

#Lighting
#ser = lc('COM5', 38400, timeout=1)
#ser.close()
#ser.open()

#ESP32
ser2 = serial.Serial('COM4', 38400, timeout=0)
ser2.close()
ser2.open()

#ser.disable()
#time.sleep(3)
#ser.enable()

cnt = 0
while (True):
    ret, frame = img.read()
    cnt = cnt + 1
    # imgg = cv2.flip(frame,1)
    imgg = frame
    imggg= frame
    # Image directory
    directory = r'D:\Sample picture 24-June-21\4_Machine Deep Learning REV-D\sample\Q23'
    # Change the current directory
    # to specified directory
    os.chdir(directory)
    # Filename
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H_%M_%S_.jpg")
    filename = timestampStr

    # Using cv2.imwrite() method
    # Saving the image
    #cv2.imwrite(filename, imgg)

    recbox, label, configuration = _cvlib.detect_common_objects(imggg,confidence=0.25, model='yolov4-tiny')
    output_image = draw_bbox(imggg, recbox, label, configuration)


    if imgg is not None:
        img_resize = cv2.resize(imgg, sizeTarget)  # resize image
        image_array = np.asarray(img_resize)  # convert image to array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1  # normalized image

        dataObj[0] = normalized_image_array  # get frist dimention
        prediction = list(model.predict(dataObj)[0])  # change np.ndarray to list
        idx = prediction.index(max(prediction))  # get index is maximun value

        dd = str(round(prediction[idx] * 100, 2)) + "%\r\n"
        ser2.write(dd.encode('utf-8'))

        if prediction[idx] * 100 > 95:
            if idx == 0:
                cv2.putText(imgg, "MT2: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 1:
                cv2.putText(imgg, "BT1: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 2:
                cv2.putText(imgg, "R248: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 3:
                cv2.putText(imgg, "SW: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 4:
                cv2.putText(imgg, "P2: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 5:
                cv2.putText(imgg, "P4: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 6:
                cv2.putText(imgg, "R128: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 7:
                cv2.putText(imgg, "J2: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 8:
                cv2.putText(imgg, "J4: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 9:
                cv2.putText(imgg, "L9: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 10:
                cv2.putText(imgg, "C200: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 11:
                cv2.putText(imgg, "L1: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 12:
                cv2.putText(imgg, "C201: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 13:
                cv2.putText(imgg, "U4: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 14:
                cv2.putText(imgg, "MT1: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 15:
                cv2.putText(imgg, "J5: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 16:
                cv2.putText(imgg, "Q10: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 17:
                cv2.putText(imgg, "Y3: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 18:
                cv2.putText(imgg, "D20: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 19:
                cv2.putText(imgg, "U19: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 20:
                cv2.putText(imgg, "Q22: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 100), 8, cv2.FILLED)
            elif idx == 21:
                cv2.putText(imgg, "Q23: " + str(round(prediction[idx] * 100, 2)) + "%", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 100), 8, cv2.FILLED)

    cv2.imshow("Predict Result", imgg)  # image show
    cv2.imshow("Predict ", imggg)  # image show
    c = cv2.waitKey(1)
    if c == 27:
        break

#ser.disable()
ser2.close()
img.release()
cv2.destroyAllWindows()
