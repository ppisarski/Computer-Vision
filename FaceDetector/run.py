import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 480)  # set width of the frame
cap.set(4, 640)  # set height of the frame


def video_detector():
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            '/Users/pawel/PycharmProjects/Computer-Vision/env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(
            '/Users/pawel/PycharmProjects/Computer-Vision/env/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            print("Found {} faces".format(len(faces)))
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Get Face
                face_img = image[y:y + h, h:h + w]
                face_gray = gray[y:y + h, h:h + w]

                eyes = eye_cascade.detectMultiScale(face_gray)
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                cv2.imshow('frame', image)
                # 0xFF is a hexadecimal constant which is 11111111 in binary.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    video_detector()
