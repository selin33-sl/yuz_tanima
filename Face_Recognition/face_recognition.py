import dlib
import cv2
import face_recognition
import sys

cap = cv2.VideoCapture(0)

# çözünürlük ayarları
cap.set(3, 640)
cap.set(4, 480)

# yüz tespiti için

detector = dlib.get_frontal_face_detector()

# Yüzünü tanımasını istediğim profillerin tanıtılması
steve=face_recognition.load_image_file("steve_jobs.jpg")

steve_encoding = face_recognition.face_encodings(steve)[0]

elon=face_recognition.load_image_file("elon_musk.jpg")

elon_encoding = face_recognition.face_encodings(elon)[0]

# Döngüyle her görüntü incelenir.

while True:
    # Kameradan alınan görüntü okunur
    _, frame = cap.read()
    face_locations = []

    faces = detector(frame)

    # Yüz tespit ettikten sonra sınırlama çizgilerinin tanımı
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        face_locations.append([y, w, h, x])

    # tespit edilmiş yüzler encoding edilir
    faces_encodigs = face_recognition.face_encodings(frame, face_locations)

    i = 0

    # encoding edilen yüzler üzerinde gezilir
    for face in faces_encodigs:
        y, w, h, x = face_locations[i]
        i += 1

        # önce bilinen yüzü sonra bilinmeyen yüzü değer atadık

        result1 = face_recognition.compare_faces([steve_encoding], face)
        result2 = face_recognition.compare_faces([elon_encoding], face)

        # Belirlenen yüz resmi aktif görüntü içerisindeki lokasyonda gösterilmek için çerçeve oluşturulur.

        if result1[0] == True:
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, h), (w, h + 30), [0, 0, 255], -1)
            cv2.putText(frame, "steve jobs", (x, h + 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        elif result2[0] == True:
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, h), (w, h + 30), [0, 0, 255], -1)
            cv2.putText(frame, "elon musk", (x, h + 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        else:
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, h), (w, h + 30), [0, 0, 255], -1)
            cv2.putText(frame, "bilinmiyor", (x, h + 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Bulunan yüz video üzerinde gösterilir

    cv2.imshow("frame", frame)

    # Videoyu kapatmak için klavyeden ‘q’ ya basılması gerekir

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

# kamera serbest bırakılır.

cap.release()

# tüm pencereleri kapatır.

cv2.destroyAllWindows()

print("yüz tanıma sistemi sorunsuz çalıştı.")
