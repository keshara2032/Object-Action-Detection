import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic # Holistic model




def extract_keypoints(results):
    if(len(results.multi_hand_landmarks) > 1):
      lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks[0] else np.zeros(21*3)
      rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[1].landmark]).flatten() if results.multi_hand_landmarks[1] else np.zeros(21*3)
    else:
      lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks[0] else np.zeros(21*3)
      rh = np.zeros(21*3)
    return np.concatenate([ lh, rh])
  
def save_landmarks(landmarks,  path, intervention, trial):

  df_path = os.path.join(path, intervention, trial)
  print(df_path)
  if not os.path.exists(df_path):
    os.makedirs(df_path)
    
  # print(landmarks)
  
  if(intervention == 'oxygen-bvm'):
    label = ['I0']*len(landmarks)
  elif(intervention == 'defib-pads'):
    label = ['I1']*len(landmarks)
    
    
  df = pd.DataFrame(landmarks)
  num_strings = ["string" + str(i) for i in range(1, 11)]  # Adjust the range as needed

  df = df['landmarks'].apply(lambda x: pd.Series(x, index=["landmark" + str(i) for i in range(1, 127)]))
  df['label'] = label
  # print(df)
  
  df.to_csv(f'{df_path}/landmarks.csv')




    # assign directory
directory = './datasets/oxygen-bvm'
# directory = './datasets/defib_pads_attachment'
intervention = 'oxygen-bvm'
# intervention = 'defib-pads'

root_path = f'./processed_data/'


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=2,
    min_tracking_confidence=0.5) as hands:
  
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


  # model = holistic
  model = hands
    


  # iterate over files in
  # that directory
  for trial,filename in enumerate(glob.iglob(f'{directory}/**/newvideo.avi', recursive=True)):
  
    print('video uri:',filename)
    
    trial = f'trial_{trial}'
    # For webcam input:
    cap = cv2.VideoCapture(filename)
    fps =  cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)

    frame_num = 0
    
    all_landmarks = []
    
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
      if(intervention == 'defib-pads'):
        image = cv2.rotate(image, cv2.ROTATE_180)
        
      results = model.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      
      if results.multi_hand_landmarks:
        
        landmarks = extract_keypoints(results=results)
        all_landmarks.append({'frame':frame_num,'landmarks':landmarks})
        
        
        hands_features = None
        for hand_landmarks in results.multi_hand_landmarks:
          hands_features = hand_landmarks.landmark
          # print(hands_features[0].x)
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', image)
      
      frame_num += 1
      
      if cv2.waitKey(delay-20) & 0xFF == ord('q'):
        break
      
    cap.release()

    save_landmarks(all_landmarks, root_path, intervention, trial)
    
    