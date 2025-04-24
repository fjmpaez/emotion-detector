from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import csv
import argparse



def extract(source, class_name, output):
    cap = cv2.VideoCapture(source)
    fmd = FaceMeshDetector(maxFaces=1)

    columns = ['Class']
    for val in range(1, 468 + 1):
        columns += ['x{}'.format(val), 'y{}'.format(val)]


    # Create the CSV file and write the header if it doesn't exist
    try:
        with open(output, 'r') as f:
            pass
    except FileNotFoundError:
        with open(output, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(columns)
    
    while cap.isOpened():
        _rt, frame = cap.read()

        if not _rt or frame is None: 
            print("No se pudo leer el frame o el video ha terminado.")
            break
        
        frame = cv2.resize(frame, (720, 480))
        _img, faces = fmd.findFaceMesh(frame)

        if faces:
            face = faces[0]
            face_data = list(np.array(face).flatten())
            face_data.insert(0, class_name)

            with open(output, 'a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(face_data)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data from video.")
    
    parser.add_argument('--source', type=str, default=0, help='Path or Url to the video file.')
    parser.add_argument('--output', type=str, default='data.csv', help='Output CSV file name.')
    parser.add_argument('--class_name', type=str, default='happy', help='Class name for the training data.')
    

    args = parser.parse_args()
    sourcepath = args.source
    class_name = args.class_name
    outputpath = args.output



    extract(sourcepath, class_name, outputpath)