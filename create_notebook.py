#!/usr/bin/env python3
"""
Script to create the movie genre prediction Jupyter notebook.
"""

import json

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Movie Genre Prediction System\n\n"
                "This notebook demonstrates a comprehensive machine learning system for predicting movie genres based on plot summaries using various feature extraction techniques and classifiers.\n\n"
                "## Features\n"
                "- **Text Preprocessing**: Advanced NLP preprocessing with NLTK\n"
                "- **Feature Extraction**: TF-IDF and Count Vectorization\n"
                "- **Multiple Classifiers**: Naive Bayes, Logistic Regression, SVM, Random Forest\n"
                "- **Evaluation**: Comprehensive model comparison and visualization\n"
                "- **Prediction Interface**: Easy-to-use prediction system"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Setup and Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages (run this cell first if packages are not installed)\n"
                "# !pip install -r movie_genre_requirements.txt"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import the movie genre prediction system\n"
                "from movie_genre_predictor import MovieGenrePredictor\n\n"
                "# Additional imports for notebook-specific functionality\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from sklearn.metrics import classification_report, confusion_matrix\n"
                "import warnings\n"
                "warnings.filterwarnings('ignore')\n\n"
                "# Set up plotting style\n"
                "plt.style.use('default')\n"
                "sns.set_palette(\"husl\")\n\n"
                "print(\"✅ All imports successful!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Initialize and Load Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize the predictor\n"
                "predictor = MovieGenrePredictor()\n"
                "print(\"🎬 Movie Genre Predictor initialized!\")\n\n"
                "# Load sample data\n"
                "print(\"📊 Loading sample data...\")\n"
                "data = predictor.load_sample_data()\n"
                "print(f\"Loaded {len(data)} movie samples\")\n"
                "print(f\"Genres: {sorted(data['genre'].unique())}\")\n\n"
                "# Display sample data\n"
                "data.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Run Complete Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run the complete analysis using the main function\n"
                "from movie_genre_predictor import main\n"
                "main()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Interactive Predictions"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the saved model for interactive use\n"
                "interactive_predictor = MovieGenrePredictor()\n"
                "interactive_predictor.load_model('movie_genre_model.pkl')\n\n"
                "def predict_genre_interactive(plot_text):\n"
                "    \"\"\"Interactive function for predicting movie genre.\"\"\"\n"
                "    prediction = interactive_predictor.predict_genre(plot_text)\n"
                "    \n"
                "    print(f\"📝 Plot: {plot_text}\")\n"
                "    print(f\"🎭 Predicted Genre: {prediction['predicted_genre']}\")\n"
                "    \n"
                "    if prediction['probabilities']:\n"
                "        print(f\"📊 Top 3 Probabilities:\")\n"
                "        top_3 = sorted(prediction['probabilities'].items(), \n"
                "                      key=lambda x: x[1], reverse=True)[:3]\n"
                "        for genre, prob in top_3:\n"
                "            print(f\"  {genre}: {prob:.3f}\")\n"
                "    \n"
                "    return prediction\n\n"
                "# Test with custom plots\n"
                "test_plots = [\n"
                "    \"A detective investigates mysterious murders in a small town\",\n"
                "    \"Aliens invade Earth and humanity must fight back\",\n"
                "    \"A couple falls in love on a cruise ship\",\n"
                "    \"Friends get into hilarious situations during their vacation\"\n"
                "]\n\n"
                "for plot in test_plots:\n"
                "    predict_genre_interactive(plot)\n"
                "    print(\"-\" * 50)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook
with open('movie_genre_prediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Jupyter notebook created successfully!")