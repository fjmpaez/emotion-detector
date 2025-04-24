# Emotion Detector

This project allows you to detect emotions from videos using machine learning models. Below are the steps to use the scripts included in this repository.

## 1. Export Training Data

The `export_training_data.py` script is used to extract facial landmarks from videos and save them as training data in a CSV file.

Parameters:
--source: Path or URL of the input video.
--output: Name of the CSV file where the extracted data will be saved. If does not exist, it creates a new one. If file exists, the information is appended
--class_name: The class label associated with the data (e.g., happy, sad).

### Command:
```bash
python src/export_training_data.py --source <video_path_or_url> --output <output_file.csv> --class_name <class_name>
```

Besides, the extract_data.sh script automates the process of extracting training data from multiple free video URLs from pexels.com for different emotion classes. It iterates over predefined classes and their associated video URLs, calling the export_training_data.py script for each video.

Parameters:
<output_file>: The path to the CSV file where the extracted data will be saved. If the file does not exist, it will be created. If it exists, new data will be appended.

```bash
bash extract_data.sh <output_file>
```

## 2. Create models

The create_model.py script is used to train a machine learning model using the data extracted from videos. The trained model is then saved to a file for later use in emotion detection.

Parameters:
--data: Path to the CSV file containing the training data. This file should include the facial landmarks and their associated class labels.
--output: Path to the file where the trained model will be saved.
--model_type: The type of model to train. Options are:
logistic_regression: A simple logistic regression model.
random_forest: A more complex random forest model.

### Command:
```bash
python src/create_model.py --data <training_data.csv> --output <model_output.pkl> --model_type <model_type>
```

## 3. Detect emotions.
The detect_emotions.py script is used to detect emotions in a video using a pre-trained machine learning model. The script processes the video frame by frame and outputs a new video with the detected emotions.

Parameters:
--source: Path or URL of the input video to process.
--output: Path to the file where the output video with detected emotions will be saved.
--model: Path to the pre-trained model file (e.g., .pkl) that will be used for emotion detection.

### Command:
```bash
python src/detect_emotions.py --source <video_path_or_url> --output <output_video.mp4> --model <trained_model.pkl>
```

## 4. Data used for the example
### Training videos
#### Happy label
- https://videos.pexels.com/video-files/6706926/6706926-hd_1920_1080_25fps.mp4
- https://videos.pexels.com/video-files/5495175/5495175-uhd_2732_1440_25fps.mp4
- https://videos.pexels.com/video-files/5536129/5536129-uhd_1440_2560_25fps.mp4
- https://videos.pexels.com/video-files/4584807/4584807-uhd_2560_1440_25fps.mp4
- https://videos.pexels.com/video-files/7976476/7976476-uhd_2732_1440_25fps.mp4


#### Sad label
- https://videos.pexels.com/video-files/5496775/5496775-uhd_2560_1440_30fps.mp4
- https://videos.pexels.com/video-files/5981354/5981354-uhd_2732_1440_25fps.mp4
- https://videos.pexels.com/video-files/6722759/6722759-uhd_2732_1440_25fps.mp4
- https://videos.pexels.com/video-files/8410107/8410107-hd_1920_1080_25fps.mp4
- https://videos.pexels.com/video-files/4494789/4494789-hd_1920_1080_30fps.mp4


### Testing videos.
#### Happy
- https://videos.pexels.com/video-files/4584804/4584804-uhd_2560_1440_25fps.mp4   
- https://videos.pexels.com/video-files/3249935/3249935-uhd_2560_1440_25fps.mp4
- https://videos.pexels.com/video-files/17160857/17160857-hd_1920_1080_30fps.mp4

#### Sad
- https://videos.pexels.com/video-files/8164443/8164443-hd_1920_1080_30fps.mp4
- https://videos.pexels.com/video-files/7280180/7280180-uhd_2560_1440_25fps.mp4
- https://videos.pexels.com/video-files/6603377/6603377-uhd_2732_1440_25fps.mp4