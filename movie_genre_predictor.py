#!/usr/bin/env python3
"""
Movie Genre Prediction System
============================

A comprehensive machine learning system for predicting movie genres based on plot summaries
and other textual information using various feature extraction techniques and classifiers.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import warnings
import pickle
import json
from typing import List, Dict, Tuple, Any
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
import os
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

class MovieGenrePredictor:
    """
    A comprehensive movie genre prediction system using multiple ML approaches.
    """
    
    def __init__(self):
        """Initialize the predictor with default settings."""
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.label_encoder = None
        self.multilabel_binarizer = None
        self.models = {}
        self.feature_extractors = {}
        self.is_multilabel = False
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK datasets."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
    
    def preprocess_text(self, text: str, use_stemming: bool = False, use_lemmatization: bool = True) -> str:
        """
        Preprocess text data with various cleaning techniques.
        
        Args:
            text (str): Input text to preprocess
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Create sample movie data for demonstration purposes.
        
        Returns:
            pd.DataFrame: Sample movie dataset
        """
        # Sample movie data with plot summaries and genres
        sample_data = [
            {
                'title': 'The Dark Knight',
                'plot': 'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy. With the help of Lieutenant Jim Gordon and District Attorney Harvey Dent, Batman sets out to destroy organized crime in Gotham.',
                'genre': 'Action'
            },
            {
                'title': 'Titanic',
                'plot': 'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
                'genre': 'Romance'
            },
            {
                'title': 'The Conjuring',
                'plot': 'Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse.',
                'genre': 'Horror'
            },
            {
                'title': 'Toy Story',
                'plot': 'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boys room.',
                'genre': 'Animation'
            },
            {
                'title': 'The Hangover',
                'plot': 'Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night and the bachelor missing.',
                'genre': 'Comedy'
            },
            {
                'title': 'Inception',
                'plot': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                'genre': 'Sci-Fi'
            },
            {
                'title': 'The Notebook',
                'plot': 'A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom, but they are soon separated because of their social differences.',
                'genre': 'Romance'
            },
            {
                'title': 'Avengers: Endgame',
                'plot': 'After the devastating events of Infinity War, the Avengers assemble once more to reverse Thanos actions and restore balance to the universe.',
                'genre': 'Action'
            },
            {
                'title': 'It',
                'plot': 'In the summer of 1989, a group of bullied kids band together to destroy a shape-shifting monster, which disguises itself as a clown and preys on the children of Derry.',
                'genre': 'Horror'
            },
            {
                'title': 'Finding Nemo',
                'plot': 'After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.',
                'genre': 'Animation'
            },
            {
                'title': 'Superbad',
                'plot': 'Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry.',
                'genre': 'Comedy'
            },
            {
                'title': 'Interstellar',
                'plot': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanitys survival.',
                'genre': 'Sci-Fi'
            },
            {
                'title': 'Mad Max: Fury Road',
                'plot': 'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners.',
                'genre': 'Action'
            },
            {
                'title': 'The Exorcist',
                'plot': 'When a teenage girl is possessed by a mysterious entity, her mother seeks the help of two priests to save her daughter.',
                'genre': 'Horror'
            },
            {
                'title': 'Shrek',
                'plot': 'A mean lord exiles fairytale creatures to the swamp of a grumpy ogre, who must go on a quest and rescue a princess for the lord in order to get his land back.',
                'genre': 'Animation'
            },
            {
                'title': 'Anchorman',
                'plot': 'Ron Burgundy is San Diegos top-rated newsman in the male-dominated broadcasting of the 1970s, but thats all about to change for Ron and his cronies when an ambitious woman is hired as a new anchor.',
                'genre': 'Comedy'
            },
            {
                'title': 'Blade Runner 2049',
                'plot': 'Young Blade Runner K discovers a long-buried secret that has the potential to plunge whats left of society into chaos.',
                'genre': 'Sci-Fi'
            },
            {
                'title': 'Casablanca',
                'plot': 'A cynical American expatriate struggles to decide whether or not he should help his former lover and her fugitive husband escape French Morocco.',
                'genre': 'Romance'
            },
            {
                'title': 'John Wick',
                'plot': 'An ex-hit-man comes out of retirement to track down the gangsters that took everything from him.',
                'genre': 'Action'
            },
            {
                'title': 'A Quiet Place',
                'plot': 'In a post-apocalyptic world, a family is forced to live in silence while hiding from monsters with ultra-sensitive hearing.',
                'genre': 'Horror'
            }
        ]
        
        # Create additional synthetic data to increase dataset size
        extended_data = []
        genre_templates = {
            'Action': [
                'A skilled agent must stop terrorists from destroying the city.',
                'Special forces team embarks on dangerous mission behind enemy lines.',
                'Martial arts expert seeks revenge against those who killed his family.',
                'Police detective chases dangerous criminal through urban landscape.',
                'Superhero battles villain threatening to destroy the world.'
            ],
            'Romance': [
                'Two people from different worlds fall deeply in love.',
                'Childhood friends reconnect and discover their true feelings.',
                'Star-crossed lovers fight against all odds to be together.',
                'Woman torn between two men must choose her true love.',
                'Couple separated by circumstances find their way back to each other.'
            ],
            'Horror': [
                'Group of friends encounter supernatural terror in abandoned house.',
                'Family moves to new home only to discover dark secrets.',
                'Campers in remote woods face unknown evil lurking in shadows.',
                'Small town plagued by mysterious disappearances and strange occurrences.',
                'Ancient curse awakens to terrorize modern world.'
            ],
            'Comedy': [
                'Mismatched partners forced to work together on important mission.',
                'Ordinary person gets caught up in extraordinary circumstances.',
                'Group of friends embark on hilarious adventure.',
                'Workplace comedy about eccentric employees and their daily mishaps.',
                'Family reunion brings together colorful relatives with comedic results.'
            ],
            'Sci-Fi': [
                'Humanity discovers alien life and must decide how to respond.',
                'Time traveler attempts to change past events with unexpected consequences.',
                'Advanced AI becomes self-aware and questions its purpose.',
                'Space exploration team encounters mysterious phenomena on distant planet.',
                'Future society grapples with advanced technology and its implications.'
            ],
            'Animation': [
                'Young hero embarks on magical quest to save their kingdom.',
                'Talking animals learn valuable lessons about friendship and courage.',
                'Unlikely companions team up for epic adventure.',
                'Child discovers magical world hidden within ordinary surroundings.',
                'Toys come to life when humans are not around.'
            ]
        }
        
        # Generate additional samples
        for genre, templates in genre_templates.items():
            for i, template in enumerate(templates):
                extended_data.append({
                    'title': f'{genre} Movie {i+1}',
                    'plot': template,
                    'genre': genre
                })
        
        # Combine original and extended data
        all_data = sample_data + extended_data
        
        return pd.DataFrame(all_data)
    
    def prepare_features(self, data: pd.DataFrame, text_column: str = 'plot') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features using multiple extraction techniques.
        
        Args:
            data (pd.DataFrame): Input dataset
            text_column (str): Name of the text column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: TF-IDF features and Count features
        """
        # Preprocess text
        processed_texts = data[text_column].apply(
            lambda x: self.preprocess_text(x, use_lemmatization=True)
        )
        
        # TF-IDF Features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        # Count Features
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            count_features = self.count_vectorizer.fit_transform(processed_texts)
        else:
            count_features = self.count_vectorizer.transform(processed_texts)
        
        return tfidf_features, count_features
    
    def prepare_labels(self, data: pd.DataFrame, label_column: str = 'genre') -> np.ndarray:
        """
        Prepare labels for training.
        
        Args:
            data (pd.DataFrame): Input dataset
            label_column (str): Name of the label column
            
        Returns:
            np.ndarray: Encoded labels
        """
        labels = data[label_column]
        
        # Check if multilabel (genres separated by commas or pipes)
        if any(isinstance(label, str) and (',' in label or '|' in label) for label in labels):
            self.is_multilabel = True
            # Split multilabel genres
            label_lists = []
            for label in labels:
                if isinstance(label, str):
                    genres = [g.strip() for g in re.split('[,|]', label)]
                    label_lists.append(genres)
                else:
                    label_lists.append([str(label)])
            
            if self.multilabel_binarizer is None:
                self.multilabel_binarizer = MultiLabelBinarizer()
                encoded_labels = self.multilabel_binarizer.fit_transform(label_lists)
            else:
                encoded_labels = self.multilabel_binarizer.transform(label_lists)
        else:
            self.is_multilabel = False
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                encoded_labels = self.label_encoder.fit_transform(labels)
            else:
                encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels
    
    def train_models(self, X_tfidf: np.ndarray, X_count: np.ndarray, y: np.ndarray):
        """
        Train multiple classification models.
        
        Args:
            X_tfidf (np.ndarray): TF-IDF features
            X_count (np.ndarray): Count features
            y (np.ndarray): Labels
        """
        print("Training models...")
        
        # Define base models
        base_models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='linear', random_state=42, probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Wrap models for multilabel if needed
        if self.is_multilabel:
            models = {name: MultiOutputClassifier(model) for name, model in base_models.items()}
        else:
            models = base_models
        
        # Train models with different feature sets
        feature_sets = {
            'tfidf': X_tfidf,
            'count': X_count
        }
        
        for feature_name, X in feature_sets.items():
            print(f"\nTraining with {feature_name} features...")
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...")
                
                try:
                    model.fit(X, y)
                    self.models[f"{model_name}_{feature_name}"] = model
                    
                    # Calculate cross-validation score
                    if not self.is_multilabel:
                        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                        print(f"    CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    
                except Exception as e:
                    print(f"    Error training {model_name}: {str(e)}")
        
        # Store feature extractors
        self.feature_extractors = {
            'tfidf': self.tfidf_vectorizer,
            'count': self.count_vectorizer
        }
        
        print("Model training completed!")
    
    def evaluate_models(self, X_tfidf: np.ndarray, X_count: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_tfidf (np.ndarray): TF-IDF test features
            X_count (np.ndarray): Count test features
            y (np.ndarray): True labels
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results
        """
        print("\nEvaluating models...")
        
        feature_sets = {
            'tfidf': X_tfidf,
            'count': X_count
        }
        
        results = {}
        
        for model_name, model in self.models.items():
            feature_type = model_name.split('_')[-1]
            X = feature_sets[feature_type]
            
            try:
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate accuracy
                if self.is_multilabel:
                    # For multilabel, use hamming loss or exact match ratio
                    from sklearn.metrics import hamming_loss
                    accuracy = 1 - hamming_loss(y, y_pred)
                else:
                    accuracy = accuracy_score(y, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                print(f"{model_name}: Accuracy = {accuracy:.3f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'accuracy': 0.0, 'error': str(e)}
        
        return results
    
    def predict_genre(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """
        Predict genre for a given text.
        
        Args:
            text (str): Input text
            model_name (str): Specific model to use (optional)
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if not self.models:
            raise ValueError("No models trained. Please train models first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text, use_lemmatization=True)
        
        # Use best model if not specified
        if model_name is None:
            model_name = list(self.models.keys())[0]  # Use first available model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        feature_type = model_name.split('_')[-1]
        
        # Extract features
        if feature_type == 'tfidf':
            features = self.tfidf_vectorizer.transform([processed_text])
        else:
            features = self.count_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)[0]
                if self.is_multilabel:
                    # For multilabel, we need to handle differently
                    probabilities = proba
                else:
                    class_names = self.label_encoder.classes_
                    probabilities = dict(zip(class_names, proba))
            except:
                probabilities = None
        
        # Decode prediction
        if self.is_multilabel:
            predicted_genres = self.multilabel_binarizer.inverse_transform([prediction])[0]
            result = {
                'predicted_genres': list(predicted_genres),
                'model_used': model_name,
                'probabilities': probabilities
            }
        else:
            predicted_genre = self.label_encoder.inverse_transform([prediction])[0]
            result = {
                'predicted_genre': predicted_genre,
                'model_used': model_name,
                'probabilities': probabilities
            }
        
        return result
    
    def visualize_results(self, data: pd.DataFrame, results: Dict[str, Dict[str, float]]):
        """
        Create visualizations for the results.
        
        Args:
            data (pd.DataFrame): Original dataset
            results (Dict[str, Dict[str, float]]): Model evaluation results
        """
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Movie Genre Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Genre distribution
        genre_counts = data['genre'].value_counts()
        axes[0, 0].bar(genre_counts.index, genre_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Genre Distribution in Dataset')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Model performance comparison
        model_names = [name for name in results.keys() if 'accuracy' in results[name]]
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = axes[0, 1].bar(range(len(model_names)), accuracies, color=colors, alpha=0.7)
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Word cloud for all plots
        all_plots = ' '.join(data['plot'].fillna(''))
        processed_plots = self.preprocess_text(all_plots, use_lemmatization=True)
        
        if processed_plots:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(processed_plots)
            axes[1, 0].imshow(wordcloud, interpolation='bilinear')
            axes[1, 0].set_title('Most Common Words in Plot Summaries')
            axes[1, 0].axis('off')
        
        # 4. Feature importance (for the best performing model)
        if model_names and accuracies:
            best_model_name = model_names[np.argmax(accuracies)]
            best_model = self.models[best_model_name]
            
            # Try to get feature importance
            feature_importance = None
            feature_names = None
            
            if 'tfidf' in best_model_name:
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
            else:
                feature_names = self.count_vectorizer.get_feature_names_out()
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_') and not self.is_multilabel:
                feature_importance = np.abs(best_model.coef_).mean(axis=0)
            
            if feature_importance is not None and feature_names is not None:
                # Get top 15 features
                top_indices = np.argsort(feature_importance)[-15:]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = feature_importance[top_indices]
                
                axes[1, 1].barh(range(len(top_features)), top_importance, color='lightcoral', alpha=0.7)
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features)
                axes[1, 1].set_title(f'Top Features ({best_model_name})')
                axes[1, 1].set_xlabel('Importance')
            else:
                axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('movie_genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'movie_genre_analysis.png'")
    
    def save_model(self, filepath: str):
        """
        Save the trained model and components.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'models': self.models,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'label_encoder': self.label_encoder,
            'multilabel_binarizer': self.multilabel_binarizer,
            'is_multilabel': self.is_multilabel,
            'feature_extractors': self.feature_extractors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.count_vectorizer = model_data['count_vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.multilabel_binarizer = model_data['multilabel_binarizer']
        self.is_multilabel = model_data['is_multilabel']
        self.feature_extractors = model_data['feature_extractors']
        
        print(f"Model loaded from {filepath}")

def main():
    """
    Main function to demonstrate the movie genre prediction system.
    """
    print("🎬 Movie Genre Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MovieGenrePredictor()
    
    # Load sample data
    print("\n📊 Loading sample data...")
    data = predictor.load_sample_data()
    print(f"Loaded {len(data)} movie samples")
    print(f"Genres: {sorted(data['genre'].unique())}")
    
    # Display sample data
    print("\n📋 Sample data:")
    print(data.head())
    
    # Prepare features and labels
    print("\n🔧 Preparing features...")
    X_tfidf, X_count = predictor.prepare_features(data)
    y = predictor.prepare_labels(data)
    
    print(f"TF-IDF features shape: {X_tfidf.shape}")
    print(f"Count features shape: {X_count.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data
    print("\n🔄 Splitting data...")
    X_tfidf_train, X_tfidf_test, X_count_train, X_count_test, y_train, y_test = train_test_split(
        X_tfidf, X_count, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    print("\n🤖 Training models...")
    predictor.train_models(X_tfidf_train, X_count_train, y_train)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    results = predictor.evaluate_models(X_tfidf_test, X_count_test, y_test)
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x].get('accuracy', 0))
    print(f"\n🏆 Best model: {best_model} (Accuracy: {results[best_model]['accuracy']:.3f})")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    test_plots = [
        "A group of superheroes must save the world from an alien invasion.",
        "Two people meet by chance and fall in love despite their differences.",
        "A family is haunted by supernatural forces in their new home.",
        "A robot becomes self-aware and questions its existence.",
        "Friends go on a hilarious road trip with unexpected adventures."
    ]
    
    for i, plot in enumerate(test_plots, 1):
        try:
            prediction = predictor.predict_genre(plot, best_model)
            if predictor.is_multilabel:
                genres = ', '.join(prediction['predicted_genres'])
                print(f"\n{i}. Plot: {plot[:60]}...")
                print(f"   Predicted genres: {genres}")
            else:
                print(f"\n{i}. Plot: {plot[:60]}...")
                print(f"   Predicted genre: {prediction['predicted_genre']}")
                
                if prediction['probabilities']:
                    top_3 = sorted(prediction['probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
                    print(f"   Top 3 probabilities: {top_3}")
        except Exception as e:
            print(f"\n{i}. Error predicting: {str(e)}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    try:
        predictor.visualize_results(data, results)
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    # Save model
    print("\n💾 Saving model...")
    predictor.save_model('movie_genre_model.pkl')
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Collect more movie data for better accuracy")
    print("2. Experiment with different preprocessing techniques")
    print("3. Try advanced models like neural networks")
    print("4. Add more features like cast, director, year, etc.")

if __name__ == "__main__":
    main()