# Multimodal Hate Speech Detection
## Project Overview
This project aims to detect hate speech through a multimodal approach, utilizing textual and visual data. The goal is to analyze social media content, identifying potential hate speech by assessing images' combined context and corresponding texts.

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

1. **Model A (LSTM & Inception v3 using Concatenation)**:
   - Utilizes LSTM and Inception v3.
   - Uses Concatenation as the fusion technique.

2. **Model B**:
   - Utilizs SBERT and DeiT.
   - Uses Concatenation as the fusion technique.

3. **Model C**:
   - Utilizes LSTM and DeiT.
   - Multimodal Infomax is the fusion technique used.
   - Integrates features from both modalities using a fusion layer to make a unified prediction.

### Training and Validation
- **Training Loop**: Involves batch processing, computation of loss (e.g., cross-entropy loss), and optimizer steps (e.g., Adam or SGD).
- **Validation**: Periodic evaluation on a validation set for performance monitoring and overfitting prevention.

### Evaluation
- **Metrics**: Models are evaluated using F1 score and ROC AUC score.
- **Model Performance Comparison**: 
   - Model A showed an accuracy of 85.6%, with strengths in textual analysis but limitations in understanding the context of images.
   - Model B achieved 72.31% accuracy, excelling in image interpretation but lacking textual context understanding.
   - Model C, the fusion model, outperformed both with an accuracy of 87.3%, effectively combining textual and visual cues.

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

