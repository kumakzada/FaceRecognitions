import cv2
import face_recognition as fr
import numpy as np
import os

#Resimlerin img degiskenlerine atandi #Basladi
img1=fr.load_image_file('Image/alon_mask_01.jpg') #resim Okunma
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2=fr.load_image_file('Image/alon_mask_02.jpg')
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB) #Resimi gray rengine donusturme

#Bitti

#Yuz Konumu ve Ozellikleri Belirleme --baslangic--
faceLoc=fr.face_locations(img1)[0] #FaceLocation degiskeni ile yuzun konumunu belirledik
#print(faceLoc) #Ekrana Yazdirdigimizda  yuzun konumlarini bize veriyor.(242, 386, 428, 201)

faceEnc=fr.face_encodings(img1)[0] #faceEnconding ile yuzun Ozelliklerini belirledik.
#print(faceEnc)

cv2.rectangle(img1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  # top, right, bottom, left
# --resim1 icin bitis--

faceLoc2=fr.face_locations(img2)[0]
faceEnc2=fr.face_encodings(img2)[0]
cv2.rectangle(img2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)
#-- resim 2 icin bitis yeri--

# Resimleri Karsilastirma Yani Bir birine benziyor mu diye --Baslangic--
result=fr.compare_faces([faceEnc], faceEnc2) # burada yaptigimiz is COMPARE_FACES ile resim ve resim2 karsilastiriliyor.
faceDis=fr.face_distance([faceEnc], faceEnc2) # burada eslesme olasiligini bulmak icin FACE_DISTANCE kullaniyoruz.
cv2.putText(img2,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255 ),2)
print(result,faceDis)    #Ekrana Yazdirmak ICin


#Ekranda Gonruntuleme
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)