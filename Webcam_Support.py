#%% Loading import and model
import keras
import numpy as np
import matplotlib as plt
from keras.models import load_model
classifier=load_model('Chubby_CelebA_model_1stGoodGen.h5')
#%% Function to resize image for the model
from keras.preprocessing import image
def process_image(image_name):
    #test_image=image.load_img(image_name,target_size=(64,64))
    test_image=image.img_to_array(image_name).astype('float32')
    test_image=np.expand_dims(test_image,axis=0)
    return test_image

# %% Live prediction feed of off webcam
import cv2 

cam = cv2.VideoCapture(0)

cv2.namedWindow("Chubby Detector")

font                   = cv2.FONT_HERSHEY_SIMPLEX

fontColor              = (0,255,0)
fontScale              = 3
thickness              = 3
lineType               = 2

img_counter = 0

fakeFrame=np.zeros((512,512,3), np.uint8)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    result=classifier.predict(process_image(cv2.resize(frame, (64,64))))

    cv2.putText(frame, str(result), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.imshow("Main frame thing", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed to quit
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()
# %%
