# No Sleep Driver
## Models
Facial landmark predictor is implemented in file facial_landmark.py. Run python facial_landmark.py to train the model.
Closed eye detector is implemented in file eye.py. Run python eye.py to train the model.
Yawn detector is implemented in file yawn.py. Run python yawn.py to train the model.
Pretrained model is stored in the folder pretrained and can be loaded using function load_pretrained_facial_landmark(), load_pretrained_eye() and load_pretrained_yawn() in each respective python file.

## Test on streaming video
Run python main.py path/to/video to test the drowsiness detection on a video.

## Using Yolo Model:
The **"UsingYolov4_All_In_One"** folder contains everything about using Yolo to detect drowsiness.

The dataset and trained_weight are too large to upload on github, therefore, I compress them in .zip files and devide them into many parts. For using, just download all the parts and extract them normally.

For source code, I use the Jupyter notebook, you can find this file in the same folder and open it with Google Colab. Any instruction and note are contained inside the notebook file.
