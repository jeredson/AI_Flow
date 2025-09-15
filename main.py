import tkinter as tk
from tkinter import Label, Button, filedialog, messagebox, Text, Frame
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI not available. Code generation will be disabled.")

class AIFlowPlatform:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Flow: Low-Code ML Platform")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Global variables
        self.selected_file = None
        self.selected_component = None
        self.trained_model = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        
        # Configure Gemini API (replace with your API key)
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except:
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create main frames
        self.create_toolbox()
        self.create_workspace()
        self.create_insight_panel()
        
    def create_toolbox(self):
        # Toolbox Frame
        self.toolbox = Frame(self.root, bg="#2c3e50", width=250)
        self.toolbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Toolbox Title
        title_label = Label(self.toolbox, text="ML Components", 
                           font=("Arial", 14, "bold"), 
                           bg="#2c3e50", fg="white")
        title_label.pack(pady=10)
        
        # ML Component Buttons
        components = [
            "Linear Regression",
            "Logistic Regression", 
            "Image Classification",
            "Text Classification"
        ]
        
        for component in components:
            btn = Button(self.toolbox, text=component,
                        font=("Arial", 10),
                        width=20, height=2,
                        bg="#3498db", fg="white",
                        command=lambda c=component: self.select_component(c))
            btn.pack(pady=5, padx=10)
    
    def create_workspace(self):
        # Workspace Frame
        self.workspace = Frame(self.root, bg="white", relief="sunken", bd=2)
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Workspace Title
        title_label = Label(self.workspace, text="Workspace", 
                           font=("Arial", 16, "bold"), bg="white")
        title_label.pack(pady=10)
        
        # Status Labels
        self.component_label = Label(self.workspace, 
                                   text="Selected Component: None",
                                   font=("Arial", 12), bg="white")
        self.component_label.pack(pady=5)
        
        self.file_label = Label(self.workspace,
                               text="File Selected: None", 
                               font=("Arial", 12), bg="white")
        self.file_label.pack(pady=5)
        
        # Input Data Button
        self.input_btn = Button(self.workspace, text="Input Data",
                               font=("Arial", 12), bg="#27ae60", fg="white",
                               command=self.select_file, width=15, height=2)
        self.input_btn.pack(pady=20)
        
        # Execute Button
        self.execute_btn = Button(self.workspace, text="Execute",
                                 font=("Arial", 14, "bold"), 
                                 bg="#e74c3c", fg="white",
                                 command=self.execute, width=15, height=2)
        self.execute_btn.pack(pady=10)
        
        # Results Frame
        self.results_frame = Frame(self.workspace, bg="white")
        self.results_frame.pack(pady=20, fill="both", expand=True)
    
    def create_insight_panel(self):
        # Insight Panel Frame
        self.insight = Frame(self.root, bg="#ecf0f1")
        self.insight.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Insight Title
        title_label = Label(self.insight, text="AI Insights", 
                           font=("Arial", 14, "bold"), bg="#ecf0f1")
        title_label.pack(pady=10)
        
        # Code Generation Button
        self.code_btn = Button(self.insight, text="Generate Code",
                              font=("Arial", 10), bg="#9b59b6", fg="white",
                              command=self.generate_code, width=20)
        self.code_btn.pack(pady=5)
        
        # Explanation Button
        self.explain_btn = Button(self.insight, text="Generate Explanation",
                                 font=("Arial", 10), bg="#8e44ad", fg="white",
                                 command=self.generate_explanation, width=20)
        self.explain_btn.pack(pady=5)
        
        # Text Display Area
        self.insight_text = Text(self.insight, wrap="word", font=("Arial", 10),
                               bg="white", height=30)
        self.insight_text.pack(pady=10, padx=10, fill="both", expand=True)
    
    def select_component(self, component):
        self.selected_component = component
        self.component_label.config(text=f"Selected Component: {component}")
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"File Selected: {filename}")
    
    def execute(self):
        if not self.selected_component:
            messagebox.showwarning("Warning", "Please select an ML component!")
            return
        
        if self.selected_component == "Image Classification":
            self.image_classification()
        elif not self.selected_file:
            messagebox.showwarning("Warning", "Please select a dataset file!")
            return
        elif self.selected_component == "Linear Regression":
            self.linear_regression()
        elif self.selected_component == "Logistic Regression":
            self.logistic_regression()
        elif self.selected_component == "Text Classification":
            self.text_classification()
    
    def linear_regression(self):
        try:
            # Load dataset
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            # Basic preprocessing
            data = data.dropna()
            
            # Assume last column is target, rest are features
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            # Handle categorical data
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.trained_model = LinearRegression()
            self.trained_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = self.trained_model.predict(X_test_scaled)
            r2 = r2_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Linear Regression Model Trained!\nR² Score: {r2:.4f}")
            
            # Create prediction interface
            self.create_prediction_interface("regression")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def logistic_regression(self):
        try:
            # Load dataset
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            # Basic preprocessing
            data = data.dropna()
            
            # Assume last column is target, rest are features
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            # Handle categorical data
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target if categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.trained_model = LogisticRegressionCV(cv=5, random_state=42)
            self.trained_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = self.trained_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Logistic Regression Model Trained!\nAccuracy: {accuracy:.4f}")
            
            # Create prediction interface
            self.create_prediction_interface("classification")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def text_classification(self):
        try:
            # Load dataset
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            # Assume first column is text, second is label
            X = data.iloc[:, 0].astype(str)
            y = data.iloc[:, 1]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create pipeline with TF-IDF and Naive Bayes
            self.trained_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            # Train model
            self.trained_model.fit(X_train, y_train)
            
            # Evaluate
            predictions = self.trained_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Text Classification Model Trained!\nAccuracy: {accuracy:.4f}")
            
            # Create text input interface
            self.create_text_prediction_interface()
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def image_classification(self):
        messagebox.showinfo("Image Classification", 
                          "Image Classification module requires pre-trained models.\n" +
                          "Please upload a pre-trained model file or implement custom training.")
    
    def create_prediction_interface(self, model_type):
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Prediction Interface
        pred_label = Label(self.results_frame, text="Make Predictions", 
                          font=("Arial", 14, "bold"), bg="white")
        pred_label.pack(pady=10)
        
        # Input fields for each feature
        self.input_vars = {}
        for i, col in enumerate(self.feature_columns):
            frame = Frame(self.results_frame, bg="white")
            frame.pack(pady=2, fill="x", padx=20)
            
            Label(frame, text=f"{col}:", font=("Arial", 10), bg="white").pack(side="left")
            var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=var, font=("Arial", 10))
            entry.pack(side="right", padx=10)
            self.input_vars[col] = var
        
        # Predict button
        predict_btn = Button(self.results_frame, text="Predict",
                           font=("Arial", 12), bg="#f39c12", fg="white",
                           command=lambda: self.make_prediction(model_type))
        predict_btn.pack(pady=20)
        
        # Result display
        self.prediction_result = Label(self.results_frame, text="", 
                                     font=("Arial", 12, "bold"), bg="white")
        self.prediction_result.pack(pady=10)
    
    def create_text_prediction_interface(self):
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Text Input Interface
        pred_label = Label(self.results_frame, text="Text Classification", 
                          font=("Arial", 14, "bold"), bg="white")
        pred_label.pack(pady=10)
        
        Label(self.results_frame, text="Enter text to classify:", 
              font=("Arial", 10), bg="white").pack(pady=5)
        
        self.text_input = Text(self.results_frame, height=5, width=50)
        self.text_input.pack(pady=10)
        
        predict_btn = Button(self.results_frame, text="Classify Text",
                           font=("Arial", 12), bg="#f39c12", fg="white",
                           command=self.classify_text)
        predict_btn.pack(pady=10)
        
        self.text_result = Label(self.results_frame, text="", 
                               font=("Arial", 12, "bold"), bg="white")
        self.text_result.pack(pady=10)
    
    def make_prediction(self, model_type):
        try:
            # Get input values
            input_data = []
            for col in self.feature_columns:
                value = self.input_vars[col].get()
                if not value:
                    messagebox.showwarning("Warning", f"Please enter value for {col}")
                    return
                input_data.append(float(value))
            
            # Scale input
            input_scaled = self.scaler.transform([input_data])
            
            # Make prediction
            prediction = self.trained_model.predict(input_scaled)[0]
            
            if model_type == "regression":
                self.prediction_result.config(text=f"Predicted Value: {prediction:.4f}")
            else:
                self.prediction_result.config(text=f"Predicted Class: {prediction}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def classify_text(self):
        try:
            text = self.text_input.get("1.0", tk.END).strip()
            if not text:
                messagebox.showwarning("Warning", "Please enter text to classify")
                return
            
            prediction = self.trained_model.predict([text])[0]
            self.text_result.config(text=f"Classification: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def generate_code(self):
        if not self.gemini_model:
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, "Gemini API not available. Please install google-generativeai and configure API key.")
            return
        
        if not self.selected_component:
            messagebox.showwarning("Warning", "Please select a component first!")
            return
        
        try:
            prompt = f"""Generate clean, commented Python code for {self.selected_component} 
            using scikit-learn. Include data loading, preprocessing, model training, 
            evaluation, and prediction. Make it educational and well-documented."""
            
            response = self.gemini_model.generate_content(prompt)
            
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, response.text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Code generation failed: {str(e)}")
    
    def generate_explanation(self):
        if not self.gemini_model:
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, "Gemini API not available. Please install google-generativeai and configure API key.")
            return
        
        if not self.selected_component:
            messagebox.showwarning("Warning", "Please select a component first!")
            return
        
        try:
            prompt = f"""Explain {self.selected_component} in machine learning. 
            Include: how it works, when to use it, advantages/disadvantages, 
            and a step-by-step explanation of the algorithm. Make it beginner-friendly."""
            
            response = self.gemini_model.generate_content(prompt)
            
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, response.text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Explanation generation failed: {str(e)}")
    
    def run(self):
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    app = AIFlowPlatform()
    app.run()