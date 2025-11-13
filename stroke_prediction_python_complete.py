
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


DATASET_PATH = "D:\\PROJECTS\\MLLab\\healthcare-dataset-stroke-data.csv"


ENABLE_OVERSAMPLING = True
OVERSAMPLE_METHOD = 'random'  # 'random' (default) or 'smote' (not implemented)
ENABLE_THRESHOLD_TUNING = True
# Default threshold (useful when ENABLE_THRESHOLD_TUNING = True)
THRESHOLD = 0.3

class StrokeDataPreprocessor:
    def __init__(self, dataset_path=None):
        """dataset_path: optional path to CSV file. If provided the loader will
        attempt to read from this path first."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.dataset_path = dataset_path
        
    def load_and_preprocess_data(self):
        """Load and preprocess the stroke dataset"""
        # If a dataset path was provided, try that first
        if self.dataset_path:
            try:
                df = pd.read_csv(self.dataset_path)
                print(f"Dataset loaded from '{self.dataset_path}' with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
            except Exception as e:
                print(f"Failed to load dataset from '{self.dataset_path}': {e}")
                print("Falling back to default/local paths and remote URL...")

        
    
    def preprocess(self, df):
        """Preprocess the data"""
        # Create a copy
        data = df.copy()
        
        # Display initial info
        print("\nInitial Data Info:")
        print(f"Missing values:\n{data.isnull().sum()}")
        print(f"\nStroke distribution:\n{data['stroke'].value_counts()}")
        print(f"Stroke percentage: {data['stroke'].mean():.3%}")
        
        # Handle missing values in bmi
        data['bmi'].fillna(data['bmi'].median(), inplace=True)
        
        # Encode categorical variables
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        # Drop id column as it's not useful for prediction
        if 'id' in data.columns:
            data.drop('id', axis=1, inplace=True)
        
        print(f"\nAfter preprocessing - Shape: {data.shape}")
        return data
    
    def prepare_features(self, data):
        """Prepare features and target variable"""
        X = data.drop('stroke', axis=1)
        y = data['stroke']
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """Scale the features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

class BaseClassifier:
    """Base class for all classifiers to ensure consistent interface"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.train_time = None
        self.predict_time = None
        
    def train(self, X_train, y_train):
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X_test):
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, y_true, y_pred, y_prob=None):
        """Evaluate model performance"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None
        }
        return metrics

# Specific classifier implementations
class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("Support Vector Machine")
        self.model = SVC(probability=True, random_state=42, class_weight='balanced')
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class RandomForestClassifierWrapper(BaseClassifier):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class DecisionTreeClassifierWrapper(BaseClassifier):
    def __init__(self):
        super().__init__("Decision Tree")
        self.model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class KNNClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("K-Nearest Neighbors")
        self.model = KNeighborsClassifier()
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class GradientBoostingClassifierWrapper(BaseClassifier):
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingClassifier(random_state=42)
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = GaussianNB()
    
    def train(self, X_train, y_train):
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        import time
        start_time = time.time()
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        self.predict_time = time.time() - start_time
        return self.predictions

class ModelComparator:
    """Class to compare multiple models and visualize results"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, model):
        """Add a model to the comparator"""
        self.models[model.name] = model
        
    def train_all_models(self, X_train, y_train):
        """Train all added models"""
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.train(X_train, y_train)
            
    def evaluate_all_models(self, X_test, y_test, threshold=None):
        """Evaluate all models and store results.

        If `threshold` is provided and a model supports `predict_proba`, use
        that threshold to produce binary predictions from the positive-class
        probabilities. Otherwise fall back to the model's `predict()` method.
        """
        import time
        print("\nEvaluating models...")
        for name, model in self.models.items():
            print(f"Evaluating {name}...")

            # If a threshold is provided and the underlying sklearn model has
            # predict_proba, use that to derive predictions at the given cutpoint.
            if threshold is not None and hasattr(model.model, 'predict_proba'):
                start = time.time()
                probs = model.model.predict_proba(X_test)[:, 1]
                preds = (probs >= threshold).astype(int)
                model.probabilities = probs
                model.predictions = preds
                model.predict_time = time.time() - start
                predictions = preds
            else:
                # Use the model's predict wrapper (which also sets probabilities
                # and timing where available).
                predictions = model.predict(X_test)

            metrics = model.evaluate(y_test, predictions, model.probabilities)
            self.results[name] = {
                'metrics': metrics,
                'predictions': predictions,
                'probabilities': model.probabilities,
                'train_time': model.train_time,
                'predict_time': model.predict_time,
                'classification_report': classification_report(y_test, predictions, output_dict=True)
            }
    
    def plot_comparison(self,y_test):
        """Create two independent figures:
        1) Accuracy vs AUC (bar chart)
        2) Precision vs Recall (precision-recall curves)

        Each figure is created separately so it appears in its own window.
        """
        model_names = list(self.results.keys())

        # --- Figure 1: Accuracy vs AUC ---
        accuracies = [self.results[name]['metrics']['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['metrics']['roc_auc'] if self.results[name]['metrics']['roc_auc'] is not None else 0.0 for name in model_names]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', alpha=0.9)
        plt.bar(x + width/2, auc_scores, width, label='AUC', color='lightcoral', alpha=0.9)
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Accuracy vs AUC Score')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
        for i, v in enumerate(auc_scores):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.show()

        # --- Figure 2: Precision vs Recall (bar chart) ---
        # Create a side-by-side bar chart showing Precision vs Recall per model.
        precisions = [self.results[name]['metrics']['precision'] for name in model_names]
        recalls = [self.results[name]['metrics']['recall'] for name in model_names]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width/2, precisions, width, label='Precision', color='seagreen', alpha=0.9)
        plt.bar(x + width/2, recalls, width, label='Recall', color='orange', alpha=0.9)
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Precision vs Recall (Bar Chart)')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(precisions):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
        for i, v in enumerate(recalls):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self):
        """Print detailed results for all models"""
        print("\n" + "="*100)
        print("DETAILED MODEL COMPARISON RESULTS")
        print("="*100)
        
        # Create results table
        results_data = []
        for name in self.models.keys():
            results_data.append({
                'Model': name,
                'Accuracy': f"{self.results[name]['metrics']['accuracy']:.4f}",
                'Precision': f"{self.results[name]['metrics']['precision']:.4f}",
                'Recall': f"{self.results[name]['metrics']['recall']:.4f}",
                'F1-Score': f"{self.results[name]['metrics']['f1_score']:.4f}",
                'AUC': f"{self.results[name]['metrics']['roc_auc']:.4f}",
                'Train Time (s)': f"{self.results[name]['train_time']:.4f}",
                'Predict Time (s)': f"{self.results[name]['predict_time']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print("\nPerformance Summary:")
        print(results_df.to_string(index=False))
        
        # Print best models by different metrics
        print("\n" + "-"*50)
        print("BEST MODELS BY METRIC:")
        print("-"*50)
        
        best_accuracy = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['accuracy'])
        best_auc = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['roc_auc'])
        best_f1 = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['f1_score'])
        best_recall = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['recall'])
        fastest_train = min(self.results.keys(), key=lambda x: self.results[x]['train_time'])
        
        print(f"Best Accuracy: {best_accuracy} ({self.results[best_accuracy]['metrics']['accuracy']:.4f})")
        print(f"Best AUC: {best_auc} ({self.results[best_auc]['metrics']['roc_auc']:.4f})")
        print(f"Best F1-Score: {best_f1} ({self.results[best_f1]['metrics']['f1_score']:.4f})")
        print(f"Best Recall: {best_recall} ({self.results[best_recall]['metrics']['recall']:.4f})")
        print(f"Fastest Training: {fastest_train} ({self.results[fastest_train]['train_time']:.4f}s)")

def print_feature_importance(comparator, feature_names):
    """Print feature importance for tree-based models"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Tree-based Models)")
    print("="*80)
    
    for name, model in comparator.models.items():
        if hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n{name} Feature Importance:")
            for i in range(len(feature_names)):
                print(f"  {i+1:2d}. {feature_names[indices[i]]:20}: {importances[indices[i]]:.4f}")


def prompt_user_instance(preprocessor, feature_columns, reference_df):
    """Prompt the user for a single example, encode categorical features using
    the preprocessor.label_encoders and return a single-row DataFrame with
    the same column order as feature_columns.

    reference_df is used to provide reasonable defaults (column means).
    """
    values = {}
    for col in feature_columns:
        # Categorical columns that were label-encoded
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            classes = list(le.classes_)
            # Present options to the user with indices
            print(f"\nSelect value for '{col}':")
            for i, c in enumerate(classes):
                print(f"  {i}. {c}")

            while True:
                raw = input(f"Enter index or exact value for '{col}' (default 0): ").strip()
                if raw == '':
                    choice = classes[0]
                    break
                # try integer index
                if raw.isdigit():
                    idx = int(raw)
                    if 0 <= idx < len(classes):
                        choice = classes[idx]
                        break
                # try exact string match (case-insensitive)
                matches = [c for c in classes if c.lower() == raw.lower()]
                if matches:
                    choice = matches[0]
                    break
                print("Invalid input. Please enter a valid index or exact option from the list.")

            # transform to integer label
            encoded = int(le.transform([choice])[0])
            values[col] = encoded
        else:
            # Numeric feature: provide mean from reference_df as default
            default = float(reference_df[col].mean())
            while True:
                raw = input(f"Enter numeric value for '{col}' (default {default:.2f}): ").strip()
                if raw == '':
                    val = default
                    break
                try:
                    val = float(raw)
                    break
                except ValueError:
                    print("Invalid number. Please try again.")
            values[col] = val

    # Build DataFrame with the same column order
    user_df = pd.DataFrame([values], columns=feature_columns)
    return user_df

def main():
    """Main function to run the complete stroke prediction framework"""
    
    print("STROKE PREDICTION FRAMEWORK")
    print("="*50)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    if DATASET_PATH:
        print(f"Using dataset path set in code: '{DATASET_PATH}'")
    preprocessor = StrokeDataPreprocessor(dataset_path=DATASET_PATH)
    df = preprocessor.load_and_preprocess_data()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Preprocess data
    data = preprocessor.preprocess(df)
    
    # Prepare features and target
    X, y = preprocessor.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optionally perform oversampling on the training set before scaling
    if ENABLE_OVERSAMPLING:
        print("\nOversampling enabled: performing random upsampling of minority class in training set...")
        # Combine X_train and y_train for easy resampling
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        majority = train_df[train_df['stroke'] == 0]
        minority = train_df[train_df['stroke'] == 1]

        if OVERSAMPLE_METHOD == 'random':
            minority_upsampled = resample(minority,
                                          replace=True,
                                          n_samples=len(majority),
                                          random_state=42)
            train_upsampled = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
            X_train = train_upsampled.drop('stroke', axis=1)
            y_train = train_upsampled['stroke']
            print(f"After upsampling - training set shape: {X_train.shape}, positive cases: {y_train.sum()}")
        else:
            print(f"Oversample method '{OVERSAMPLE_METHOD}' not implemented. Skipping oversampling.")

    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"\nData split:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Positive cases in training: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Positive cases in test: {y_test.sum()} ({y_test.mean():.2%})")
    
    # Step 2: Initialize models and comparator
    print("\nStep 2: Initializing models...")
    comparator = ModelComparator()
    
    # Add models - This is where you can easily add/remove algorithms
    comparator.add_model(LogisticRegressionClassifier())
    comparator.add_model(SVMClassifier())
    comparator.add_model(RandomForestClassifierWrapper())
    comparator.add_model(DecisionTreeClassifierWrapper())
    comparator.add_model(KNNClassifier())
    comparator.add_model(GradientBoostingClassifierWrapper())
    comparator.add_model(NaiveBayesClassifier())
    
    print(f"Total models to compare: {len(comparator.models)}")
    
    # Step 3: Train and evaluate all models
    print("\nStep 3: Training and evaluating models...")
    print(f"\nConfiguration: ENABLE_OVERSAMPLING={ENABLE_OVERSAMPLING}, ENABLE_THRESHOLD_TUNING={ENABLE_THRESHOLD_TUNING}, THRESHOLD={THRESHOLD}")
    comparator.train_all_models(X_train_scaled, y_train)
    comparator.evaluate_all_models(X_test_scaled, y_test, threshold=(THRESHOLD if ENABLE_THRESHOLD_TUNING else None))

    # --- DEBUG / DIAGNOSTICS for Random Forest ---
    # If Random Forest metrics look wrong (precision/recall == 0) this will
    # print extra info to help debug (prediction distribution, confusion
    # matrix and a small sample of actual vs predicted labels).
    rf_name = 'Random Forest'
    if rf_name in comparator.results:
        print(f"\nDiagnostic for '{rf_name}':")
        rf_res = comparator.results[rf_name]
        # Print classification report (readable)
        print("Classification report:")
        print(classification_report(y_test, rf_res['predictions']))

        # Print confusion matrix
        cm = confusion_matrix(y_test, rf_res['predictions'])
        print("Confusion matrix:")
        print(cm)

        # Distribution of predicted labels
        unique, counts = np.unique(rf_res['predictions'], return_counts=True)
        print("Prediction distribution:")
        for u, c in zip(unique, counts):
            print(f"  Predicted {u}: {c} samples")

        # Show a small sample of (y_true, y_pred)
        print("Sample true vs predicted (first 30):")
        sample_true = np.array(y_test)[:30]
        sample_pred = np.array(rf_res['predictions'])[:30]
        for i, (t, p) in enumerate(zip(sample_true, sample_pred)):
            print(f"  {i+1:02d}. true={t}  pred={p}")
    else:
        print(f"\nNo results found for '{rf_name}' to diagnose.")
    
    # Step 4: Visualize results
    print("\nStep 4: Generating visualizations...")
    comparator.plot_comparison(y_test)
    
    # Step 5: Print detailed results
    comparator.print_detailed_results()
    
    # Feature importance for tree-based models
    print_feature_importance(comparator, X.columns)

    # Step 6: Interactive single-sample prediction
    while True:
        do_input = input("\nDo you want to input a custom patient for prediction? (y/n): ").strip().lower()
        if do_input in ('y', 'n', ''):
            break
        print("Please enter 'y' or 'n'.")

    if do_input == 'y':
        feature_cols = list(X.columns)
        user_df = prompt_user_instance(preprocessor, feature_cols, X)

        # Scale the user input using the fitted scaler
        try:
            user_scaled = preprocessor.scaler.transform(user_df)
        except Exception as e:
            print(f"Error scaling input: {e}")
            return

        print("\nPredictions for the provided input:")
        for name, model in comparator.models.items():
            try:
                # Use the model's predict wrapper to ensure probabilities are set
                preds = model.predict(user_scaled)
                prob = model.probabilities[0] if model.probabilities is not None else None
            except Exception:
                # fallback to sklearn API if wrapper fails
                if hasattr(model.model, 'predict'):
                    preds = model.model.predict(user_scaled)
                else:
                    preds = [None]
                if hasattr(model.model, 'predict_proba'):
                    prob = model.model.predict_proba(user_scaled)[:, 1][0]
                else:
                    prob = None

            pred_label = preds[0] if isinstance(preds, (list, np.ndarray)) else preds
            prob_str = f"{prob:.3f}" if prob is not None else "N/A"

            # If threshold tuning is enabled, show thresholded decision too
            if ENABLE_THRESHOLD_TUNING and prob is not None:
                thresholded = int(prob >= THRESHOLD)
                print(f"- {name}: Predicted={pred_label} (prob={prob_str}), Thresholded(>={THRESHOLD})={thresholded}")
            else:
                print(f"- {name}: Predicted={pred_label} (prob={prob_str})")

    print("\nDone.")

# Template for adding new algorithms
def how_to_add_new_algorithm():
    """
    HOW TO ADD NEW ALGORITHMS:
    
    class YourNewAlgorithm(BaseClassifier):
        def __init__(self):
            super().__init__("Your Algorithm Name")
            self.model = YourSKLearnModel()  # Initialize your model
        
        def train(self, X_train, y_train):
            import time
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.train_time = time.time() - start_time
        
        def predict(self, X_test):
            import time
            start_time = time.time()
            self.predictions = self.model.predict(X_test)
            if hasattr(self.model, 'predict_proba'):
                self.probabilities = self.model.predict_proba(X_test)[:, 1]
            else:
                self.probabilities = self.predictions
            self.predict_time = time.time() - start_time
            return self.predictions
    
    Then add: comparator.add_model(YourNewAlgorithm())
    """

if __name__ == "__main__":
    main()