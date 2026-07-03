# Fake News Detection

A machine learning project for detecting fake news using deep learning and natural language processing techniques.

## Overview

This project implements a fake news classification system using neural networks with word embeddings (GloVe). The model is trained on a Kaggle fake news dataset containing news articles labeled as either genuine (0) or fake (1).

## Dataset

- **Source**: Kaggle Fake News Competition
- **Size**: 20,761 samples (after cleaning)
- **Features**: Text content from news articles
- **Target**: Binary classification (0 = Real, 1 = Fake)
- **Class Distribution**: Balanced (~50% real, ~50% fake)

### Dataset Features:
- `id`: Unique identifier for each article
- `title`: Title of the news article
- `author`: Author of the article
- `text`: Full text content of the article
- `label`: Classification label (0 for real, 1 for fake)

## Project Structure

```
Fake_news/
├── README.md
└── clean_Fake_news_code.ipynb    # Main project notebook with all analysis and modeling
```

## Technologies & Libraries

- **Data Processing**: 
  - Pandas (data manipulation)
  - NumPy (numerical computations)
  
- **Machine Learning & Deep Learning**:
  - TensorFlow/Keras (neural network implementation)
  - Scikit-learn (preprocessing and metrics)
  
- **Text Processing**:
  - Keras Tokenizer (text tokenization)
  - GloVe 6B 200D word embeddings (pre-trained word vectors)
  
- **Visualization**:
  - Matplotlib (plotting)
  
- **Data Source**:
  - Kaggle API (dataset download)

## Methodology

### 1. Data Loading & Preprocessing
- Import training data from Kaggle fake news competition
- Remove missing values
- Extract relevant columns (text and label)

### 2. Text Cleaning
- Remove URLs and special characters
- Convert to lowercase
- Remove extra whitespace

### 3. Text Tokenization & Embedding
- Use Keras Tokenizer to convert text to sequences
- Pad sequences to uniform length
- Load pre-trained GloVe 6B 200D word embeddings
- Create embedding matrix for the model

### 4. Model Training
- Build deep neural network using Keras
- Use embedding layer with GloVe weights
- Train on cleaned and processed data
- Evaluate using accuracy and confusion matrix metrics

### 5. Evaluation
- Accuracy score
- Confusion matrix analysis
- Classification metrics

## Key Features

✅ **Balanced Dataset**: Approximately equal distribution between real and fake news  
✅ **Pre-trained Embeddings**: Uses GloVe word vectors for better semantic understanding  
✅ **Deep Learning**: Neural network architecture for robust classification  
✅ **Data Cleaning**: Comprehensive text preprocessing pipeline  
✅ **Evaluation Metrics**: Multiple metrics for model assessment  

## Results

The model achieves strong classification performance on the balanced fake news dataset with the following characteristics:

- **Dataset Size**: 20,761 samples
- **Real News**: 10,387 samples
- **Fake News**: 10,374 samples
- **Embedding Dimension**: 200D (GloVe)

## Usage

### Prerequisites
- Python 3.x
- Kaggle account and credentials
- Required libraries (see Technologies section)

### Running the Notebook

1. **Setup Kaggle Credentials**:
   - Run the Kaggle login cell to authenticate
   - Ensure your Kaggle API key is configured

2. **Download Data**:
   - The notebook automatically downloads:
     - Fake News competition dataset (~46.5 MB)
     - GloVe 6B 200D embeddings (~259 MB)

3. **Run Analysis**:
   - Execute cells sequentially
   - Monitor training progress
   - Review evaluation metrics

## Model Architecture

The neural network uses:
- **Embedding Layer**: Pre-trained GloVe 200D embeddings
- **LSTM/Dense Layers**: For sequence processing and feature extraction
- **Output Layer**: Binary classification (fake/real)
- **Activation Functions**: ReLU and Sigmoid
- **Loss Function**: Binary crossentropy

## Data Source Attribution

- **Kaggle Fake News Competition**: https://www.kaggle.com/competitions/fake-news
- **GloVe Embeddings**: https://www.kaggle.com/datasets/incorpes/glove6b200d

## Future Improvements

- Experiment with different neural network architectures (CNN, Transformers)
- Incorporate additional features (metadata, publication date)
- Implement ensemble methods
- Deploy model as API
- Add explainability features (attention mechanisms)

## License

This project uses publicly available datasets from Kaggle. Please refer to Kaggle's terms of service for dataset usage.

## Author

[SutapaSusovita](https://github.com/SutapaSusovita)

## Contact

For questions or suggestions, please open an issue on the GitHub repository.


