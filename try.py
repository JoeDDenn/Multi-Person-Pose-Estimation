import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Define arm keypoints indices
LEFT_ARM_INDICES = [5, 7, 9, 11, 13, 15]
RIGHT_ARM_INDICES = [6, 8, 10, 12, 14, 16]

# Define punch detection function
def detect_punch(keypoints):
    # Check if left or right arm keypoints are visible and have high enough confidence score
    has_left_arm = all(keypoints[i, 2] > 0.2 for i in LEFT_ARM_INDICES)
    has_right_arm = all(keypoints[i, 2] > 0.2 for i in RIGHT_ARM_INDICES)
    if not (has_left_arm or has_right_arm):
        return False

    # Check if arm is raised and ready to punch
    if has_left_arm:
        shoulder, elbow, wrist = keypoints[11], keypoints[13], keypoints[15]
    else:
        shoulder, elbow, wrist = keypoints[12], keypoints[14], keypoints[16]
    if elbow[1] < shoulder[1] and elbow[0] > shoulder[0]:
        # Check if arm is thrust forward in a punch
        if wrist[1] > elbow[1] and wrist[1] - elbow[1] > abs(wrist[0] - elbow[0]) * 2:
            # Check if body is rotated and arm is thrown with force
            if shoulder[0] < elbow[0] and shoulder[1] < elbow[1]:
                return True

    return False

# Define rendering functions
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        
        # Detect punches
        if detect_punch(person):
            highlight_person(frame, person)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def highlight_person(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    min_x, max_x = int(np.min(shaped[:, 1])), int(np.max(shaped[:, 1]))
    min_y, max_y = int(np.min(shaped[:, 0])), int(np.max(shaped[:, 0]))
    
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

# Open video file
cap = cv2.VideoCapture('ufc.mp4')

# Define edges for rendering
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Process video file
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160,256)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints and detect punches
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    # Display image
    cv2.imshow('Punch Detection', frame)
    
    # Check for quit key
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()