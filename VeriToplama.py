import cv2
cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
k = cv2.waitKey(100) & 0xff 
ad = input('\n Lutfen Kişi Adını giriniz:  ')
print("\n Lutfen Kameraya bakınız ...")
print()
print()
secim=input('Lutfen bir sezim yapiniz \n s resim kayit etme')
count = 0
if secim=='s':
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("Image/" + str(ad) + '_' +  
                        str(count) + ".jpg",gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            break
        elif count >= 15: 
             break
cam.release()
cv2.destroyAllWindows()
#gray[y:y+h,x:x+w]