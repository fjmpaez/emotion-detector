import cvzone
import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import argparse


HEIGHT = 480
WIDTH = 720

emotion_colors = {
        "happy": (0, 255, 0),       # Verde
        "sad": (255, 0, 0),         # Azul
        "angry": (0, 0, 255),       # Rojo
        "scared": (255, 255, 0),    # Amarillo
        "surprised": (255, 0, 255)  # Magenta
    }

def detect(source, model, output):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video {source}")
        return
    

    # Lee el primer frame para determinar el tamaño de all_frames
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    real_frame = frame.copy()
    all_frames = cvzone.stackImages([real_frame, frame], 2, 0.70)

    output_height, output_width, _ = all_frames.shape

    output_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"avc1"), int(cap.get(cv2.CAP_PROP_FPS)), (output_width, output_height))

    if not output_video.isOpened():
        print(f"Error: Could not open video writer {output}")
        return

    
    fmd = FaceMeshDetector()

    with open(model,'rb') as f:
        data = pickle.load(f)
        faces_detector_model = data['model']
        label_encoder = data['label_encoder']
       

    # taking video frame by frame
    while cap.isOpened():
        rt,frame = cap.read()
        frame = cv2.resize(frame,(WIDTH,HEIGHT))

        real_frame = frame.copy()

        img, faces = fmd.findFaceMesh(frame)
        
        cvzone.putTextRect(real_frame, ('Original'), (10, 70), colorR=(5, 5, 5))
        cvzone.putTextRect(frame, ('Emotion: '), (10, 70), colorR=(5, 5, 5))

        if faces:
            face = faces[0]
            face_data = list(np.array(face).flatten())
        

            try:
                prediction = faces_detector_model.predict([face_data])
                emotion = label_encoder.inverse_transform(prediction)[0]
                
                color = emotion_colors.get(emotion, (255, 255, 255))  # Blanco por defecto
                cvzone.putTextRect(frame, emotion, (250, 70), colorR=color)

            except Exception as e:
                pass

        all_frames = cvzone.stackImages([real_frame,frame],2,0.70)
        cv2.imshow('frame',all_frames)
        cv2.waitKey(1)
        output_video.write(all_frames)
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect emotions from video.")
    parser.add_argument('--source', type=str, default=0, help='Path to the video file.')
    parser.add_argument('--output', type=str, help='Result output video.')
    parser.add_argument('--model', type=str, help='Model.')

    args = parser.parse_args()
    sourcepath = args.source
    outputpath = args.output
    modelpath = args.model


    detect(source=sourcepath, model=modelpath, output=outputpath)


