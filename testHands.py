### Importing mediapipe API and OpenCV
import mediapipe as mp, cv2

# Getting solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Take webcam input
cap = cv2.VideoCapture(0)


### Using MediaPipe Hands
# Exception handling
with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:

    # Running while camera is open
    while cap.isOpened():
        
        # Attempt to read image 
        success, image = cap.read()

        # Case: Empty camera frame
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image horizontally
        image = cv2.flip(image, 1)

        # Convert order of color 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enabling passing by reference
        image.flags.writeable = False

        # Using mediapipe to process the image
        results = hands.process(image)


        ### Draw the hand annotations on the image
        # Enabling drawing on image
        image.flags.writeable = True

        # Reconverting the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Drawing landmarks on images
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        
        # Show image
        cv2.imshow("MediaPipe Hands", image)

        # Repeat
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()





