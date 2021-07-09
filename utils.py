import cv2
import numpy as np
import os

from datetime import datetime

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            confidence = detection[-1]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        confidence = confidences[i]
        final_boxes.append((box, confidence))

    return final_boxes

def visualize(frame, boxes, model, label_dict):
    for box, conf in boxes: 
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        right = left + width
        bottom = top + height

        # Crop frame and run prediction 
        try: 
            img = frame[(top-20):(top+height+10),(left-10):(left+width+20)] 
            img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
            img = img / 255.
            img = img.flatten()
            
            prediction = model.predict([img])[0]
            name = label_dict[prediction]
        except:
            name = 'Detecting...'
            print('Out of frame')

        # Draw bouding box and text
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)
        text = f'{name} - {conf:.2f}'
        cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    text2 = f"Number of faces detected: {len(boxes)}"
    print(text2)
    cv2.putText(frame, text2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)


def capture(frame, name, boxes):
    # Capture multiple photos of the target in front of the webcam 
    # To use for model training 
    DATA_FOLDER = './model/data'
    folder = os.path.join(DATA_FOLDER, name)
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    for box, _ in boxes:
        try: 
            left, top, width, height = box
            crop = frame[(top-20):(top+height+10),(left-10):(left+width+20)] 
            now = datetime.now().strftime("%H%M%S")
            image_name = os.path.join(folder,f'{now}.jpg')
            cv2.imwrite(image_name, crop)
        except:
            pass

