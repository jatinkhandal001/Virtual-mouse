import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Smooth tracking variables
prev_x, prev_y = 0, 0
smoothening = 7  # Higher = smoother

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image horizontally for natural interaction
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]

            # Convert index tip position to image coordinates
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            cv2.circle(image, (x, y), 10, (255, 0, 0), -1)

            # Convert to screen coordinates
            target_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
            target_y = np.interp(index_tip.y, [0, 1], [0, screen_height])

            # Smooth cursor movement
            curr_x = prev_x + (target_x - prev_x) / smoothening
            curr_y = prev_y + (target_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Calculate gesture distances
            click_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) -
                                        np.array([thumb_tip.x, thumb_tip.y]))
            right_click_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) -
                                              np.array([ring_tip.x, ring_tip.y]))

            # Y-positions for scroll detection
            index_y = index_tip.y
            middle_y = middle_tip.y

            # Left Click
            if click_dist < 0.05:
                pyautogui.click()
                cv2.putText(image, 'Left Click', (x + 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click
            elif right_click_dist < 0.05:
                pyautogui.rightClick()
                cv2.putText(image, 'Right Click', (x + 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Scroll Up
            elif index_y < middle_y - 0.02:
                pyautogui.scroll(30)
                cv2.putText(image, 'Scroll Up', (x + 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Scroll Down
            elif middle_y < index_y - 0.02:
                pyautogui.scroll(-30)
                cv2.putText(image, 'Scroll Down', (x + 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display
    cv2.imshow("Virtual Mouse", image)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()