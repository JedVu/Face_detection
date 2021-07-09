# Face_detection

Locate and recognize human face using YOLO and OpenCV

Download YOLO model and weight at: https://drive.google.com/drive/folders/1QO9ydq_cUHlfpKK78DSUymHii5aix2jf?usp=sharing
Store the files in folder name yolo in the main directory.
To run detection run:
python3 main.py
To collect data images by capturing from webcam, create a folder name data in the model directory and run:
python3 main.py --capture True --name <target name>
Captured images will be saved in the data folder with the target name as its directory.

To train and export classifier, run:
python3 train.py
--- Note: Classifier weight is not provided. Please run train.py before the first use of main.py
