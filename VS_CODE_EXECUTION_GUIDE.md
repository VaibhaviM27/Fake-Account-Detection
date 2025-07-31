# Movie Genre Prediction - VS Code Execution Guide

## 🎬 Complete Machine Learning System for Movie Genre Prediction

This guide provides detailed steps to execute a comprehensive machine learning system that predicts movie genres based on plot summaries using TF-IDF, Count Vectorization, and multiple classifiers (Naive Bayes, Logistic Regression, SVM, Random Forest).

## 📋 What You'll Build

- **Text Preprocessing Pipeline**: Advanced NLP with NLTK
- **Feature Extraction**: TF-IDF and Count Vectorization with n-grams
- **Multiple ML Models**: 8 different model combinations
- **Evaluation System**: Cross-validation, confusion matrices, performance metrics
- **Prediction Interface**: Real-time genre prediction
- **Visualization Tools**: Performance charts, word clouds, feature importance
- **Model Persistence**: Save/load trained models

## 🚀 Quick Start (5 Minutes)

### Step 1: Setup Environment
```bash
# Create and activate virtual environment
python -m venv movie_genre_env

# Windows:
movie_genre_env\Scripts\activate
# macOS/Linux:
source movie_genre_env/bin/activate

# Install dependencies
pip install -r movie_genre_requirements.txt
```

### Step 2: Configure VS Code
1. **Open VS Code** → File → Open Folder → Select project directory
2. **Set Python Interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose your virtual environment
3. **Install Extensions**: Python (Microsoft), Jupyter (optional)

### Step 3: Test System
```bash
python test_system.py
```

### Step 4: Run Complete Analysis
```bash
python movie_genre_predictor.py
```

## 📁 Project Files

```
movie-genre-prediction/
├── movie_genre_predictor.py          # Main system (850+ lines)
├── movie_genre_requirements.txt      # Dependencies
├── test_system.py                    # Quick test script
├── create_notebook.py                # Jupyter notebook generator
├── VS_CODE_EXECUTION_GUIDE.md        # This guide
├── movie_genre_model.pkl             # Saved model (generated)
└── movie_genre_analysis.png          # Visualizations (generated)
```

## 🔧 Detailed VS Code Execution Steps

### Method 1: Run Complete System

#### Step 1: Open Main File
1. In VS Code, open `movie_genre_predictor.py`
2. Review the code structure and main components

#### Step 2: Execute Options
**Option A: Run with F5 (Debug Mode)**
- Press `F5` to run in debug mode
- Set breakpoints by clicking line numbers
- Use Debug Console to inspect variables

**Option B: Run with Ctrl+F5 (Normal Mode)**
- Press `Ctrl+F5` to run without debugging
- Faster execution, no debugging features

**Option C: Use Terminal**
- Press `Ctrl+`` (backtick) to open terminal
- Run: `python movie_genre_predictor.py`

#### Step 3: Monitor Output
The system will show progress through these stages:
```
🎬 Movie Genre Prediction System
📊 Loading sample data... (50 movies, 6 genres)
🔧 Preparing features... (TF-IDF + Count vectorization)
🤖 Training models... (8 model combinations)
📈 Evaluating models... (Performance metrics)
🔮 Testing predictions... (Sample predictions)
📊 Creating visualizations... (Charts and plots)
💾 Saving model... (Pickle file)
✅ Analysis complete!
```

### Method 2: Interactive Development

#### Step 1: Create Test File
1. Create new file: `Ctrl+N`
2. Save as `my_test.py`

#### Step 2: Import and Explore
```python
from movie_genre_predictor import MovieGenrePredictor
import pandas as pd

# Initialize system
predictor = MovieGenrePredictor()
print("System initialized!")

# Load data
data = predictor.load_sample_data()
print(f"Loaded {len(data)} movies")
print(data.head())

# Test preprocessing
sample = "A superhero saves the world from evil villains!"
processed = predictor.preprocess_text(sample)
print(f"Original: {sample}")
print(f"Processed: {processed}")
```

#### Step 3: Run Code Blocks
- **Select code** and press `Shift+Enter` to run in terminal
- **Right-click** → "Run Selection/Line in Python Terminal"
- Use **Python Interactive** window for notebook-like experience

### Method 3: Jupyter Notebook (Optional)

#### Step 1: Generate Notebook
```bash
python create_notebook.py
```

#### Step 2: Open in VS Code
1. Open `movie_genre_prediction.ipynb`
2. Select Python kernel (your virtual environment)
3. Run cells with `Shift+Enter`

## 📊 Expected Results

### Performance Metrics
- **Training Time**: 30-60 seconds
- **Accuracy Range**: 80-95%
- **Best Model**: Usually Logistic Regression + TF-IDF
- **Dataset**: 50 movies across 6 genres

### Sample Output
```
📊 Loading sample data...
Loaded 50 movie samples
Genres: ['Action', 'Animation', 'Comedy', 'Horror', 'Romance', 'Sci-Fi']

🔧 Preparing features...
TF-IDF features shape: (50, 247)
Count features shape: (50, 184)

🤖 Training models...
Training with tfidf features...
  Training naive_bayes...
    CV Accuracy: 0.850 (+/- 0.100)
  Training logistic_regression...
    CV Accuracy: 0.900 (+/- 0.080)

🏆 Best model: logistic_regression_tfidf (Accuracy: 1.000)

🔮 Testing predictions...
1. Plot: A group of superheroes must save the world...
   Predicted genre: Action
   Top 3 probabilities: [('Action', 0.856), ('Sci-Fi', 0.098)]

✅ Analysis complete!
```

### Generated Files
- `movie_genre_model.pkl`: Trained model (can be loaded later)
- `movie_genre_analysis.png`: Comprehensive visualization with 4 charts

## 🎯 Understanding the System

### Core Components

#### 1. MovieGenrePredictor Class
```python
class MovieGenrePredictor:
    def __init__(self):              # Initialize NLTK, vectorizers
    def preprocess_text(self):       # Clean and normalize text
    def load_sample_data(self):      # Create/load movie dataset
    def prepare_features(self):      # Extract TF-IDF and Count features
    def train_models(self):          # Train 8 model combinations
    def evaluate_models(self):       # Test performance
    def predict_genre(self):         # Make predictions
    def visualize_results(self):     # Create charts
    def save_model(self) / load_model(self): # Persistence
```

#### 2. Text Preprocessing Pipeline
```python
def preprocess_text(self, text):
    text = text.lower()                          # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove special chars
    tokens = word_tokenize(text)                 # Tokenize
    tokens = [t for t in tokens if t not in stopwords] # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
    return ' '.join(tokens)
```

#### 3. Feature Extraction
- **TF-IDF**: Term Frequency × Inverse Document Frequency
  - Highlights important words unique to documents
  - Parameters: max_features=5000, ngram_range=(1,2)
- **Count Vectorization**: Simple word frequency counting
  - Parameters: max_features=3000, ngram_range=(1,2)

#### 4. Machine Learning Models
- **Naive Bayes**: Fast, probabilistic, works well with text
- **Logistic Regression**: Linear model, good baseline
- **SVM**: Finds optimal decision boundary
- **Random Forest**: Ensemble of decision trees

### Model Training Process
1. **Data Split**: 80% training, 20% testing
2. **Cross-Validation**: 3-fold CV for robust evaluation
3. **Feature Combinations**: Each model trained with TF-IDF and Count features
4. **Performance Tracking**: Accuracy, confusion matrices, classification reports

## 🛠️ Customization Options

### 1. Use Your Own Data
```python
# Load your CSV file (must have 'plot' and 'genre' columns)
import pandas as pd
data = pd.read_csv('your_movies.csv')

predictor = MovieGenrePredictor()
X_tfidf, X_count = predictor.prepare_features(data)
y = predictor.prepare_labels(data)
predictor.train_models(X_tfidf, X_count, y)
```

### 2. Modify Text Preprocessing
```python
# Customize preprocessing
processed = predictor.preprocess_text(
    text, 
    use_stemming=True,      # Use stemming instead of lemmatization
    use_lemmatization=False
)
```

### 3. Add New Models
```python
from sklearn.ensemble import GradientBoostingClassifier

# Add to base_models in train_models method
base_models['gradient_boosting'] = GradientBoostingClassifier()
```

### 4. Tune Hyperparameters
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_tfidf, y)
best_model = grid_search.best_estimator_
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
**Error**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**:
```bash
# Ensure virtual environment is activated
pip install -r movie_genre_requirements.txt
```

#### 2. NLTK Data Missing
**Error**: `LookupError: Resource punkt not found`
**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

#### 3. VS Code Python Issues
**Problem**: VS Code not finding Python interpreter
**Solution**:
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose your virtual environment path
4. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

#### 4. Memory Issues
**Error**: `MemoryError` during training
**Solutions**:
- Reduce `max_features` in vectorizers (from 5000 to 1000)
- Use smaller dataset for testing
- Close other applications

#### 5. Low Performance
**Problem**: Models showing poor accuracy
**Solutions**:
- Increase dataset size (current: 50 samples)
- Improve text preprocessing
- Try different feature extraction methods
- Tune hyperparameters

### VS Code Specific Tips

#### Debug Configuration
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Movie Genre Predictor",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/movie_genre_predictor.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

#### Workspace Settings
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./movie_genre_env/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

## 📈 Performance Benchmarks

| Model | Feature Type | Typical Accuracy | Training Time |
|-------|-------------|------------------|---------------|
| Naive Bayes | TF-IDF | 85-90% | < 1 second |
| Logistic Regression | TF-IDF | 90-95% | 1-2 seconds |
| SVM | TF-IDF | 85-92% | 2-5 seconds |
| Random Forest | TF-IDF | 80-88% | 3-10 seconds |

## 🚀 Next Steps & Extensions

### Immediate Improvements (1-2 weeks)
1. **Larger Dataset**: Use IMDb, MovieLens, or TMDB data
2. **More Features**: Add cast, director, year, budget information
3. **Advanced Models**: Neural networks, transformers (BERT)
4. **Multilabel Support**: Handle movies with multiple genres

### Advanced Features (1-3 months)
1. **Web Interface**: Flask/FastAPI deployment
2. **Real-time Data**: Web scraping movie databases
3. **API Integration**: Connect to movie APIs
4. **Advanced Analytics**: Genre trends, box office correlation

### Production Deployment (3+ months)
1. **Containerization**: Docker deployment
2. **Cloud Hosting**: AWS, GCP, or Azure
3. **Monitoring**: Logging, performance tracking
4. **CI/CD Pipeline**: Automated testing and deployment

## 📋 Success Checklist

✅ **Environment Setup**
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] VS Code configured with Python interpreter

✅ **System Testing**
- [ ] `test_system.py` runs without errors
- [ ] All imports successful
- [ ] Sample data loads correctly
- [ ] Text preprocessing works

✅ **Full System Execution**
- [ ] `movie_genre_predictor.py` runs completely
- [ ] All 8 models train successfully
- [ ] Accuracy scores above 80%
- [ ] Predictions work on sample plots

✅ **Output Verification**
- [ ] `movie_genre_model.pkl` file created
- [ ] `movie_genre_analysis.png` visualization generated
- [ ] Console output shows complete analysis

✅ **Understanding**
- [ ] Can explain how TF-IDF works
- [ ] Understand model performance differences
- [ ] Can modify preprocessing parameters
- [ ] Can add custom movie plots for prediction

## 🎓 Learning Outcomes

After completing this project, you will understand:

### Machine Learning Concepts
- Text preprocessing and feature extraction
- TF-IDF vs Count Vectorization
- Model comparison and evaluation
- Cross-validation and performance metrics

### NLP Techniques
- Tokenization, lemmatization, stopword removal
- N-gram analysis
- Text classification approaches

### Python & VS Code
- Virtual environment management
- Debugging and interactive development
- Package management and dependencies
- Code organization and structure

### Practical Skills
- End-to-end ML pipeline development
- Model persistence and deployment
- Data visualization and interpretation
- Error handling and troubleshooting

---

## 🎬 Ready to Start?

1. **Quick Test**: `python test_system.py`
2. **Full Analysis**: `python movie_genre_predictor.py`
3. **Custom Predictions**: Load the saved model and try your own movie plots!

**Happy coding! This system provides a solid foundation for movie genre prediction and demonstrates professional ML development practices. 🚀✨**