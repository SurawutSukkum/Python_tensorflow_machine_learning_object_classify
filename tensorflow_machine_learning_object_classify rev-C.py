from tensorflow.keras.models import load_model
import cv2
import numpy as np
import serial

#img = cv2.imread("D:/Sample picture 24-June-21/Image.png") #path image
img = cv2.VideoCapture(0)
sizeTarget = (224, 224)
np.set_printoptions(suppress=True)
dataObj = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
model = load_model("D:/Sample picture 24-June-21/keras_model.h5") #path model

ser = serial.Serial('COM4', 38400, timeout=0)
ser.close()
ser.open()
print(ser.isOpen)    
print(ser.portstr)     
print(ser.name) 
ser.write(b'@00F01200D9\r\n@00L11D\r\n')
cnt = 0
while(True):
     ret, frame = img.read()
     #imgg = cv2.flip(frame,1)
     imgg = frame
     cnt = cnt + 1
     if cnt > 100:
        ser.write(b'@00F01200D9\r\n@00L11D\r\n')  
     if cnt >200:
        ser.write(b'@00F16800E5\r\n@00L11D\r\n')
        cnt=0
         
     if imgg is not None:
        img_resize = cv2.resize(imgg,sizeTarget) #resize image      
        image_array = np.asarray(img_resize)#convert image to array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 #normalized image

        dataObj[0] = normalized_image_array #get frist dimention 
        prediction =  list(model.predict(dataObj)[0])#change np.ndarray to list 
        idx = prediction.index(max(prediction)) #get index is maximun value

        if  prediction[idx]*100 > 90:
            if idx == 0:
                cv2.putText(imgg, "MT2: "+str(round(prediction[idx]*100,2)) +"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 1:
                cv2.putText(imgg, "BT1: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 2:
                cv2.putText(imgg, "UBLOX: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 3:
                cv2.putText(imgg, "R248: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 4:
                cv2.putText(imgg, "P3: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 5:
                cv2.putText(imgg, "SW: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 6:
                cv2.putText(imgg, "P2: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 7:
                cv2.putText(imgg, "P4: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 8:
                cv2.putText(imgg, "R128: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 9:
                cv2.putText(imgg, "J2: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 10:
                cv2.putText(imgg, "J4: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 11:
                cv2.putText(imgg, "L9: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 12:
                cv2.putText(imgg, "C200: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 13:
                cv2.putText(imgg, "L1: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 14:
                cv2.putText(imgg, "J1: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 15:
                cv2.putText(imgg, "C201: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
            elif idx == 16:
                cv2.putText(imgg, "U4: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
                                                    
     cv2.imshow("Predict Result",  imgg) #image show
     cv2.imshow("Real",  frame) #image show
     c = cv2.waitKey(1)
     if c == 27:
        break
     
ser.close()  
img.release()
cv2.destroyAllWindows()
