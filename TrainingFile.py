import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Alogrithm
# 1.Eigenfaces(1991)
# 2.LBPH(1996) (Local Binary Pattern Histogram)
# 3.FisherFace(1997)
# 4.SIFT(1999)
# 5.SURF(2006)

path = 'dataset'

def getImageWithId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    faces = []
    IDS = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImage, 'uiunt8')
        # print(faceNp)  
        Id = (os.path.split(imagePath)[-1].split('.')[1])
        # dataSet\\User.1.1.jpg 
        Id = int(Id)
        print(Id)
        faces.append(Id)
        IDS.append(Id)
        cv2.imshow("Training",faceNp)
        cv2.waitKey(1)
    return IDS,faces

IDS,faces = getImageWithId(path)
recognizer.train(faces,np.array(IDs)) #error
recognizer.write('recognizer/Training.yml')
print("Training Complete")
cv2.destroyAllwindows()    



    
