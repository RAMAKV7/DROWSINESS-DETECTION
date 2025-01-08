# DRIVER DROWSINESS-DETECTION USING YOLOV11
This project utilizes YOLO V11 (You Only Look Once) deep learning model to detect signs of driver fatigue in real-time through facial feature analysis, ensuring enhanced road safety. 
# Real-Time drowsiness Detection with YOLOv11 and PyTorch

Identification of objects in an image considered a common assignment for the human brain, though not so trivial for a machine. Identification and localization of objects in photos is a computer vision task called ‘object detection’. One of the most popular algorithms to date for real-time object detection is YOLO (You Only Look Once).

In this project, we performed drowsiness detection to check whether a person is awake or drowsy, using the latest YOLOv11 implementation.
## Prerequisites

- PyTorch
- Torchvision
- Torchaudio
- OpenCV
- Matplotlib
- Numpy
- ipywidgets
- PyQt5
- lxml

## Installation

1. Install the required dependencies by running the following command:

    !pip3 install torch torchvision torchaudio


    !pip install opencv-python matplotlib numpy ipywidgets


    !pip install pyqt5==5.15.2 lxml


2. Clone the YOLOv11 repository by running:
 
    !git clone https://github.com/ultralytics/yolov5

3. Navigate to the cloned directory:

    cd yolov11

4. Install the requirements for YOLOv11:

    !pip install -r requirements.txt


5. Clone the labelImg repository for image labeling:

    !git clone https://github.com/tzutalin/labelImg.git


## Usage

1. Collect Images:
- Connect a webcam to your computer.
- Run the code to capture images for the specified labels ("awake" and "drowsy").
- Images will be saved in the "data/images" directory.

2. Label Images:
- Open the labelImg tool to label the collected images as "awake" or "drowsy".
- Save the labeled annotations as XML files.

3. Train the YOLOv11 Model:
- Run the training script with the specified parameters (image size, batch size, epochs, data configuration, weights, etc.).
- The model will be trained on the labeled images using the YOLOv5 architecture.

4. Load the Trained Model:
- Load the trained model weights for inference.
- Perform object detection on images or video streams.

 

## Dataset

The dataset used for training can be downloaded from the following link:  
[Complete YOLO Drowsy Images Dataset](https://firebasestorage.googleapis.com/v0/b/electora-8c1d6.appspot.com/o/Complete%20YOLO%20Drowsy%20Images%20Dataset.zip?alt=media)

The dataset includes annotated images in YOLO format, with labels specifying the "awake" and "drowsy" states.

---

## Files and Structure

### Repository Structure:
```plaintext
│ notebook.ipynb               # Initial training session
│ notebook2.ipynb              # Resumed training session
├─training_and_testing_files:  # Outputs and intermediate results
│   args.yaml
│   confusion_matrix.png
│   confusion_matrix_normalized.png
│   F1_curve.png
│   labels.jpg
│   labels_correlogram.jpg
│   PR_curve.png
│   P_curve.png
│   results.csv
│   results.png
│   results1.png
│   results2.png
│   results3.png
│   results4.png
│   results5.png
│   R_curve.png
│   train_batch*.jpg
│   val_batch*_labels.jpg
│   val_batch*_pred.jpg
├─.ipynb_checkpoints:          # Checkpointed files
└─pics:                        # Selected visualization outputs
    confusion_matrix.png
    F1_curve.png
    labels.jpg
    train_batch34880.jpg
    val_batch1_pred.jpg
```

---

## Training Process Overview

1. **Dataset Loading:**
   - YOLO utilizes a `yaml` file to define the paths for training and validation datasets and their corresponding class labels.

2. **Key Parameters for Training:**
   - **Epochs:** The model was trained for 50 epochs to balance learning and avoiding overfitting.
   - **Image Size:** All input images were resized to 640x640 pixels to match YOLO's expected square input format.
   - **Batch Size:** A batch size of 16 was used to optimize training efficiency.

3. **Hardware and Framework:**
   - **Framework:** PyTorch backend via the `ultralytics` library.
   - **Hardware:** Training and inference were performed on an Nvidia RTX 3080 GPU.

---

## Dependencies and Requirements

- Python 3.8+
- Key Python libraries:
  - `ultralytics` for YOLO-based training and inference
  - `torch` and `torchvision` for PyTorch operations
  - `numpy`, `matplotlib`, and `pandas` for data processing and visualization
- GPU with CUDA support (recommended: Nvidia RTX 3080 or higher)

Install dependencies using the following command:
```bash
pip install ultralytics torch torchvision numpy matplotlib pandas
```

---

## Key Results

- **Performance Metrics:**
  - Confusion Matrix: `confusion_matrix.png` and `confusion_matrix_normalized.png`
  - Precision-Recall (PR) Curve: `PR_curve.png`
  - F1 Curve: `F1_curve.png`
  - Results stored in `results.csv` and visualized in `results*.png`

- **Visualizations:**
  - Training Batches: `train_batch*.jpg`
  - Validation Predictions: `val_batch*_pred.jpg`

---

## Citation

If you use this repository in your research, please cite it using the following DOI link:  
[DOI Placeholder]()

Feel free to open issues or pull requests if you encounter any problems or have suggestions for improvement.

