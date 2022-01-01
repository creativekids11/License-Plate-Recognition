from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd="C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

cascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states={"AN":"Andaman and Nicobar",
    "AP":"Andhra Pradesh","AR":"Arunachal Pradesh",
    "AS":"Assam","BR":"Bihar","CH":"Chandigarh",
    "DN":"Dadra and Nagar Haveli","DD":"Daman and Diu",
    "DL":"Delhi","GA":"Goa","GJ":"Gujarat",
    "HR":"Haryana","HP":"Himachal Pradesh",
    "JK":"Jammu and Kashmir","KA":"Karnataka","KL":"Kerala",
    "LD":"Lakshadweep","MP":"Madhya Pradesh","MH":"Maharashtra","MN":"Manipur",
    "ML":"Meghalaya","MZ":"Mizoram","NL":"Nagaland","OD":"Odissa",
    "PY":"Pondicherry","PN":"Punjab","RJ":"Rajasthan","SK":"Sikkim","TN":"TamilNadu",
    "TR":"Tripura","UP":"Uttar Pradesh", "WB":"West Bengal","CG":"Chhattisgarh",
    "TS":"Telangana","JH":"Jharkhand","UK":"Uttarakhand"}

def extract_num(img_filename):
    img=cv2.imread(img_filename)
    #Img To Gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4)
    #crop portion
    for (x,y,w,h) in nplate:
        wT,hT,cT=img.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate=img[y+a:y+h-a,x+b:x+w-b,:]
        #make the img more darker to identify LPR
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
        #read the text on the plate
        read=pytesseract.image_to_string(plate)
        read=''.join(e for e in read if e.isalnum())
        stat=read[0:2]
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cv2.imshow("plate",plate)
        
    cv2.imwrite("Result.png",img)
    cv2.imshow("Result",img)
    if cv2.waitKey(0)==113:
        exit()
    cv2.destroyAllWindows()

extract_num("car_img.png")

