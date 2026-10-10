import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hands = hands[0]
        lmList = hands['lmList']
        handtype = hands['type']

        for x, y, z in lmList:
            cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)

            for start, end in [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]:
                x1, y1, _ = lmList[start]
                x2, y2, _ = lmList[end]
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow('Hand Dtection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()