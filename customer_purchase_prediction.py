# customer_purchase_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")

class CustomerPurchasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.using_sample_data = False
        
    def load_and_preprocess_data(self):
        """Load and preprocess the Bank Marketing dataset"""
        try:
            # Load the dataset - using a different approach
            print("Loading dataset...")
            # Try alternative URL or local download
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
                df = pd.read_csv(url, sep=';')
            except:
                # Try the original URL again
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
                df = pd.read_csv(url, sep=';')
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Convert target variable to binary
            df['y'] = df['y'].map({'yes': 1, 'no': 0})
            
            # Separate features and target
            X = df.drop('y', axis=1)
            y = df['y']
            
            self.feature_names = X.columns.tolist()
            self.using_sample_data = False
            
            # Identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            print(f"Categorical columns: {categorical_cols}")
            print(f"Numerical columns: {numerical_cols}")
            
            # Encode categorical variables
            X_encoded = X.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # Scale numerical features
            X_encoded[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            
            return X_encoded, y, df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback: create sample data for demonstration
            print("Creating sample data for demonstration...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if unable to download from URL"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample features
        age = np.random.randint(18, 80, n_samples)
        balance = np.random.normal(1500, 800, n_samples)
        duration = np.random.randint(60, 1000, n_samples)
        campaign = np.random.randint(1, 10, n_samples)
        previous = np.random.randint(0, 10, n_samples)
        
        # Create target variable with some relationship to features
        y = ((age > 40) & (balance > 1000) & (duration > 300) & (campaign < 5) & (previous > 0)).astype(int)
        
        # Add some noise
        y = np.where(np.random.random(n_samples) < 0.2, 1 - y, y)
        
        X = pd.DataFrame({
            'age': age,
            'balance': balance,
            'duration': duration,
            'campaign': campaign,
            'previous': previous
        })
        
        # Set feature names for sample data
        self.feature_names = X.columns.tolist()
        self.using_sample_data = True
        
        return X, y, pd.DataFrame()
    
    def train_model(self, X, y):
        """Train the decision tree model"""
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Positive class ratio: {y_train.mean():.3f}")
        
        # Train initial decision tree
        print("\nTraining initial decision tree...")
        dt_classifier = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        dt_classifier.fit(X_train, y_train)
        
        # Evaluate initial model
        self.evaluate_model(dt_classifier, X_test, y_test, "Initial Model")
        
        # Hyperparameter tuning (simplified for sample data)
        print("\nPerforming hyperparameter tuning...")
        
        if self.using_sample_data:
            # Simplified parameter grid for sample data
            param_grid = {
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            }
        else:
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'criterion': ['gini', 'entropy']
            }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train with best parameters
        self.model = grid_search.best_estimator_
        self.model.fit(X_train, y_train)
        
        # Evaluate tuned model
        self.evaluate_model(self.model, X_test, y_test, "Tuned Model")
        
        return X_test, y_test
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate the model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{model_name} Performance:")
        print("=" * 50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Plot ROC curve if it's the final model
        if model_name == "Tuned Model":
            self.plot_roc_curve(y_test, y_pred_proba)
    
    def plot_confusion_matrix(self, y_test, y_pred, title):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Purchase', 'Purchase'],
                    yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is not None:
            # Ensure feature names match the model's feature importance array
            if len(self.feature_names) == len(self.model.feature_importances_):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
                plt.title('Top 10 Feature Importances')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.show()
            else:
                print("Warning: Feature names don't match feature importance array")
                print(f"Feature names length: {len(self.feature_names)}")
                print(f"Feature importance length: {len(self.model.feature_importances_)}")
    
    def visualize_decision_tree(self):
        """Visualize the decision tree"""
        if self.model is not None:
            plt.figure(figsize=(20, 12))
            plot_tree(
                self.model,
                feature_names=self.feature_names,
                class_names=['No Purchase', 'Purchase'],
                filled=True,
                rounded=True,
                proportion=True,
                max_depth=3
            )
            plt.title('Decision Tree Visualization (First 3 Levels)')
            plt.tight_layout()
            plt.show()
    
    def save_model(self, filename='customer_purchase_predictor.pkl'):
        """Save the trained model"""
        if self.model is not None:
            model_artifacts = {
                'model': self.model,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'using_sample_data': self.using_sample_data
            }
            
            joblib.dump(model_artifacts, filename)
            print(f"Model saved as {filename}")
    
    def load_model(self, filename='customer_purchase_predictor.pkl'):
        """Load a trained model"""
        try:
            model_artifacts = joblib.load(filename)
            self.model = model_artifacts['model']
            self.feature_names = model_artifacts['feature_names']
            self.label_encoders = model_artifacts['label_encoders']
            self.scaler = model_artifacts['scaler']
            self.using_sample_data = model_artifacts.get('using_sample_data', False)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print("Model file not found. Please train a model first.")

def main():
    """Main function to run the complete pipeline"""
    print("Customer Purchase Prediction using Decision Trees")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = CustomerPurchasePredictor()
    
    # Load and preprocess data
    X, y, original_df = predictor.load_and_preprocess_data()
    
    # Train the model
    X_test, y_test = predictor.train_model(X, y)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Visualize decision tree
    predictor.visualize_decision_tree()
    
    # Save the model
    predictor.save_model()
    
    print("\nPipeline completed successfully!")
    
    # Show sample predictions
    if predictor.using_sample_data:
        print("\nSample Predictions:")
        print("=" * 30)
        sample_data = X.iloc[:3].copy()
        predictions = predictor.model.predict(sample_data)
        probabilities = predictor.model.predict_proba(sample_data)
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            outcome = "Purchase" if pred == 1 else "No Purchase"
            confidence = prob[1] if pred == 1 else prob[0]
            print(f"Customer {i+1}: {outcome} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()