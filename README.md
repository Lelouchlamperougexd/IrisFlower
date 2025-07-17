# Iris Classification Project

A machine learning project that classifies iris flowers into three species using Support Vector Machine (SVM) classifier.

## Dataset

This project uses the famous Iris dataset created by R.A. Fisher in 1936. The dataset contains 150 samples of iris flowers with the following features:

- **Sepal Length** (cm)
- **Sepal Width** (cm)  
- **Petal Length** (cm)
- **Petal Width** (cm)
- **Species**: Iris-setosa, Iris-versicolor, Iris-virginica

### Dataset Statistics
- **Total Samples**: 150 (50 samples per species)
- **Features**: 4 numeric features
- **Classes**: 3 species (evenly distributed)
- **Missing Values**: None

## Project Structure

```
iris-classification/
├── iris.data              # Raw dataset file
├── iris.names             # Dataset documentation
├── classification.py      # Main classification script
├── requirements.txt       # Python dependencies
├── IrisModel.pickle       # Trained model (generated after running)
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- numpy
- matplotlib
- seaborn
- pandas
- scikit-learn

## Usage

Run the main classification script:

```bash
python classification.py
```

The script will:
1. Load and explore the dataset
2. Perform data analysis and visualization
3. Train an SVM classifier
4. Evaluate model performance
5. Save the trained model as `IrisModel.pickle`
6. Demonstrate predictions on new samples

## Features

### Data Analysis
- Dataset overview and statistical summary
- Missing value and duplicate detection
- Feature correlation analysis
- Pair plot visualization for all features

### Model Training
- Support Vector Machine (SVM) classifier
- 80/20 train-test split
- Default SVM parameters

### Model Evaluation
- Accuracy score calculation
- Detailed classification report
- Confusion matrix visualization

### Model Persistence
- Save trained model using pickle
- Load and verify saved model functionality

## Results

The SVM classifier typically achieves high accuracy on the Iris dataset due to the dataset's well-separated classes. The model performance includes:

- Accuracy metrics
- Precision, recall, and F1-score for each species
- Confusion matrix showing prediction accuracy

## Example Predictions

The script includes example predictions for new iris samples:

```python
# Example input features: [sepal_length, sepal_width, petal_length, petal_width]
new_samples = [
    [3.0, 2.0, 1.0, 0.2],    # Likely Iris-setosa
    [4.9, 2.2, 3.8, 1.1],   # Likely Iris-versicolor  
    [5.3, 2.5, 4.6, 1.9],   # Likely Iris-virginica
    [5.1, 3.5, 1.4, 0.2]    # Likely Iris-setosa
]
```

## Model vs Loaded Model

The script demonstrates the difference between the original trained model and the loaded model:

- **`model`**: The original SVM classifier trained in the current session
- **`loaded_model`**: The same model loaded from the pickle file

Both models should produce identical predictions, demonstrating successful model persistence.

## Visualizations

The project generates several visualizations:
- Pair plots showing feature relationships colored by species
- Bar chart comparing average feature values across species
- Correlation heatmap of features
- Confusion matrix heatmap
