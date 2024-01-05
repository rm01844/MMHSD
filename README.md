# MMHSD
# Multimodal Hate Speech Detection

This project implements a multimodal approach to detect hate speech, leveraging both textual and visual data. It's uniquely crafted to scrutinize social media content, identifying potential hate speech by examining the combined context of images and their corresponding text.

## Installation
Install the required Python packages and dependencies:

```bash
pip install easyocr tqdm
```

## Dataset
The project uses a combination of image and text data:

- **Image Dataset**: `img_resized.zip`
- **Text Dataset**: `img_txt.zip`
- **Ground Truth Data**: `MMHS150K_GT.json`

Setup datasets in your project directory:

```python
!unzip /path/to/img_resized.zip
!unzip /path/to/img_txt.zip
```

## Code Structure

### Data Preprocessing
The code begins with the importation of necessary libraries such as `PIL`, `torch`, `easyocr`, and `nltk`. It sets up the environment for handling both image and text data.

### Model Architecture
- **Neural Network**: The model utilizes a deep learning architecture combining convolutional neural networks (CNNs) for image processing and recurrent neural networks (RNNs) for text processing.
- **Image Processing**: Employing pre-trained models like ResNet or VGG for feature extraction from images.
- **Text Processing**: Utilizing GloVe embeddings for text representation, followed by an LSTM or GRU for capturing textual context.
- **Fusion Layer**: A layer that intelligently combines features from both modalities (text and image) to make a unified prediction.

### Training and Validation
- **Training Loop**: Details on batch processing, loss computation (e.g., cross-entropy loss), and optimizer steps (e.g., Adam or SGD).
- **Validation**: Periodic evaluation of the model on a validation set to monitor performance and avoid overfitting.

### Evaluation
- **Metrics**: Usage of F1 score and ROC AUC score for performance measurement.
- **Model Performance**: Analyzing the balance between precision and recall, understanding the model's ability to generalize across various data samples.

## Usage

### Training the Model
Run the training script with the necessary dataset paths:

```python
python train_model.py --image_data /path/to/img_resized --text_data /path/to/img_txt
```

### Evaluating the Model
Evaluate the model's performance on a test set:

```python
python evaluate_model.py --model /path/to/saved_model --test_data /path/to/test_data
```

## Results and Insights
- **Model Accuracy**: Share insights on the achieved accuracy, discussing any notable successes or challenges encountered.
- **Error Analysis**: Provide observations from the misclassified samples, which can offer insights for future improvements.
- **Graphs and Visualizations**: Include key graphs that illustrate the performance trends over different epochs or varying hyperparameters.

## Contact
Information for users to reach out with questions, contributions, or feedback.
email: raqeebmhd619@gmail.com

