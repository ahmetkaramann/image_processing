                            #RENK İLE NESNE TESPİTİ
""" Belirli renklerde bulunan nesnelerin tespitinin nasıl yapılacağını kontur bulma yöntemi ile öğrenicez.
konturlar basitçe aynı renk ve yoğunluğa sahip tüm sürekli noktaları birleştiren bir eğri olarak açıklanır.
konturlar şekil analizi ve nesne algılama ve tanıma için kullanışlı bir araçtır.
burada RGB to HSV yapıcaz Hue(ton) Saturation(doygunluk) Value(parlaklık)"""

import cv2
from collections import deque    #tespit ettiğimiz objenin merkezini depolamak için kullanılır.
import numpy as np

#Nesne Merkezi Depolama ve Pencere Boyutları:
buffer_size = 16
pts = deque(maxlen=buffer_size)

#Sarı Renk Aralığının Belirlenmesi: HSV
yellowLower = (20, 100, 100)
yellowUpper = (40, 255, 255)


#Kamera Bağlantısının Sağlanması:
cap = cv2.VideoCapture(0)
cap.set(3,960)      #genişlik (width)
cap.set(4,480)      #yükseklik (height)

#Ana Döngü (Main Loop): 
#Ana döngü, görüntü akışının sürekli olarak işlenmesini sağlar. Her döngüde bir görüntü okunur ve işlenir. success değeri, görüntünün başarılı bir şekilde okunup okunmadığını belirtir.
while True:
    success, imgOriginal = cap.read()       #kameradan kaynaklı sorun yaşadığımızda opencv herhangi bir sorun vermiyor bu satırın altına adirekt olarak komut koymamalıyız if satırları ile devam etmemiz lazım.
    if success:
        
        #blur   #detayını azaltıp noise ları dışarıda bırakmamız lazım
        blurred =cv2.GaussianBlur(imgOriginal, ksize=(11,11), sigmaX=0)
        
        #Hsv
        hsv=cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV image", hsv)
        
        #mavi için maske oluştur
        mask = cv2.inRange(hsv, yellowLower, yellowUpper) # Belirlenen mavi renk aralığında bir maske oluşturulur. Bu maske, algılanan mavi nesneyi beyaz ve diğer renkleri siyah olarak temsil eder.
        
        #maskenin etrafındaki kalan gürültüleri sil
            #bunuda erezyon ve genişleme ile yapıcaz
        
        mask = cv2.erode(mask, kernel=None, iterations=2)   #erezyon
        mask = cv2.dilate(mask, kernel=None, iterations=2)  #genişleme
        cv2.imshow("Mask+Erezyon image", mask)
        
        #Kontur Bulma ve Nesne Merkezinin Hesaplanması:
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Sadece external kullandık çünkü dışından mavi olduğunu görmemiz yeterli içine gerek yok
        
        center = None #nesnemizin merkezi ilerleyen kısımlarda doldurcaz ve deque nin içine depolayacaz
        
        if len(contours) > 0 :
            #En Büyük Konturun İşlenmesi:
            c = max(contours, key=cv2.contourArea) #cv2.contourArea: Bu, OpenCV'nin konturun alanını hesaplamak için kullandığı bir fonksiyondur. Belirtilen konturun alanını hesaplar.
                                                    #key argümanı, en büyük öğeyi bulmak için kullanılan bir işlevi belirtir. 
            #dikdörtgene çevir
            rect = cv2.minAreaRect(c)     #konturu kapsayacak min alana sahip bir rect döndürcez
            
            #Döner Dikdörtgenin İşlenmesi ve Merkezin Bulunması:
            ((x,y), (width,height), rotation) = rect
            
            s = "x:{}, y:{}, width:{}, height:{}, rotation:{}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation))
            print(s)
                
            #Kutu ve Merkezi Çizimleri:
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #Moment (görüntünün merkezini bulmamıza yarayan yapı)
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))  #bir touple içersine (x,y) yazmış olduk, (1,0)/(0,0) ve (0,1)/(0,0)                                  
            
            #konturu çizdir :mavi
            cv2.drawContours(imgOriginal, [box], 0, (255,0,0),2)
            
            #merkeze bir tane nok0ta çizelim :                   
            cv2.circle(imgOriginal, center, 5, (0,0,255),-1) #-1 demek yuvarlağın içini doldur demek
            
            #bilgileri ekrana yazdır (img, ne isteniyorsa, kordinatları, font, metin boyutu, renk, kalınlık)
            cv2.putText(imgOriginal, s, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2)
            
            
            
        # deque
        pts.appendleft(center)  #
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Orijinal Tespit",imgOriginal)    
            
             
            
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord("q") : break


        



























