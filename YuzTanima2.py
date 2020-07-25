# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:49:41 2020

@author: abdul
"""
import os
import face_recognition as fr
import numpy as np
import cv2
 
key=cv2.waitKey(1)
path='Database'

images=[] # resimleri Saklamak icin
classNames=[] #resimlerin adlarini turmak icin

myList=os.listdir(path) #burada LISTDIR ile Dosyalarin icindeki resimleri okuyoruz
print(myList) # Resim Isimlerini Yazdiriyoruz
for cl in myList:
    curImage=cv2.imread(f'{path}/{cl}') #burada resim kirpma islemi yapiyoruz
    images.append(curImage) #append ile kelme islemi gerceklesiyor
    classNames.append(os.path.splitext(cl)[0]) # resim adi ve uzantisi bir birinden ayrilip CLASSNAMES ekleniyor


    
    
    
    
def findEncodings(images):  #resim ozelliklerini alma fonksiyonu
    encodList=[] # resim ozelliklerini saklayan dizi
    for img in images: # imagelari tek tek gray donusturuyor ve yuz ozelliklerini aliyor
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(img)[0]
        encodList.append(encode)
    return encodList

#def liste(name):
   # with open('Katilim.csv','r+') as f:
    #    myDataList = f.readlines()
     #   nameList =[]
      #  for line in myDataList:
       #     entry = line.split(',')
        #    nameList.append(entry[0])
        #if name not in  line:
         #   now = datetime.now()
          #  dt_string = now.strftime("%H:%M:%S")
           # f.writelines(f'\n{name},{dt_string}')
    
    
    

encodListKnow=findEncodings(images)
print('Enconding Completed')

cap=cv2.VideoCapture(0) #video kamere acildi 
#cap=cv2.VideoCapture('video.mp4')

while True:
    secess,img=cap.read()
    imgS=cv2.resize(img,(0,0),fx=0.25,fy=0.25) # burada sistemin hizini arttirmak icin gelen goruntuyu 1/4 yapiyoruz
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    faceLocFrame=fr.face_locations(imgS)
    faceEncFrame=fr.face_encodings(imgS,faceLocFrame)
    
    for encodFace,faceLoc in zip(faceEncFrame,faceLocFrame):
        matches=fr.compare_faces(encodListKnow, encodFace)
        faceDis=fr.face_distance(encodListKnow, encodFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
       # if matches[matchIndex]:
        #        name = classNames[matchIndex].upper()
                #print(name)
          #      y1,x2,y2,x1 = faceLoc
           #     y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
             #   cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
              #  cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                
        
        if faceDis[matchIndex]< 0.50:
            name = classNames[matchIndex].upper()
           
            
        else: name = 'Unknown'
        #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        
# =============================================================================
#         if name=='Unknown':
#             if key == ord('s'):
#                 sayac=0
#                 ad=str(input('Lutefen Ad giriniz: '))
#                 for i in range(10):
#                     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#                     kirp=img[y1:y2,x2:x1]
#                     cv2.imwrite('Image/'+ad+ f'{sayac+1}'+'.jpg',kirp)
#                     sayac=+i
# =============================================================================
                    
               
    cv2.imshow('Webcam',img)
    key=cv2.waitKey(1)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()