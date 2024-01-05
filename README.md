# MMHSD
## Project Overview
This project aims to detect hate speech through a multimodal approach, utilizing both textual and visual data. The goal is to analyze social media content, identifying potential hate speech by assessing images' combined context and corresponding texts.

## Installation
Install the required Python packages and dependencies:

```bash
pip install easyocr tqdm
```

## Dataset
The project uses datasets consisting of images and text:

- **Image Dataset**: `img_resized.zip`
- **Text Dataset**: `img_txt.zip`
- **Ground Truth Data**: `MMHS150K_GT.json`

Set up the datasets in your project directory:

```python
!unzip /path/to/img_resized.zip
!unzip /path/to/img_txt.zip
```

## Code Structure

### Data Preprocessing
The code starts with importing libraries such as `PIL`, `torch`, `easyocr`, and `nltk`, setting up the environment for handling image and text data.

### Model Architectures
The project explores three distinct model architectures:

1. **Model A (Text-Based Model)**:
   - Utilizes GloVe embeddings for text representation followed by an LSTM/GRU layer for capturing textual context.
   - Aimed at understanding the textual content of the posts.

2. **Model B (Image-Based Model)**:
   - Employs a CNN architecture like ResNet or VGG for image feature extraction.
   - Focuses on analyzing the visual content.

3. **Model C (Fusion Model)**:
   - A combination of both text and image models.
   - Integrates features from both modalities using a fusion layer to make a unified prediction.

### Training and Validation
- **Training Loop**: Involves batch processing, computation of loss (e.g., cross-entropy loss), and optimizer steps (e.g., Adam or SGD).
- **Validation**: Periodic evaluation on a validation set for performance monitoring and overfitting prevention.

### Evaluation
- **Metrics**: Models are evaluated using F1 score and ROC AUC score.
- **Model Performance Comparison**: 
   - Model A showed an accuracy of X%, with strengths in textual analysis but limitations in understanding the context of images.
   - Model B achieved Y% accuracy, excelling in image interpretation but lacking in textual context understanding.
   - Model C, the fusion model, outperformed both with an accuracy of Z%, effectively combining textual and visual cues.

## Results and Insights
- **Accuracy and Analysis**: Discuss the accuracy of each model, highlighting their strengths and weaknesses.
- **Error Analysis**: Insights from misclassified samples to guide future improvements.
- **Graphs and Visualizations**: Key graphs illustrating performance trends and comparisons between models.

## Usage

### Training the Model
```python
python train_model.py --image_data /path/to/img_resized --text_data /path/to/img_txt
```

### Evaluating the Model
```python
python evaluate_model.py --model /path/to/saved_model --test_data /path/to/test_data
```

## Contact
For questions, contributions, or feedback, reach out at raqeebmhd619@gmail.com.
