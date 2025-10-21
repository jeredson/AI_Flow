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

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available. Code generation will be disabled.")

class AIFlowPlatform:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Flow: Low-Code ML Platform")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        self.selected_file = None
        self.selected_component = None
        self.trained_model = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        
        if GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key="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")
            except:
                self.groq_client = None
        else:
            self.groq_client = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.create_toolbox()
        self.create_workspace()
        self.create_insight_panel()
        
    def create_toolbox(self):
        self.toolbox = Frame(self.root, bg="#2c3e50", width=250)
        self.toolbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        title_label = Label(self.toolbox, text="ML Components", 
                           font=("Arial", 14, "bold"), 
                           bg="#2c3e50", fg="white")
        title_label.pack(pady=10)
        
        components = ["Linear Regression", "Logistic Regression", "Image Classification", "Text Classification"]
        
        for component in components:
            btn = Button(self.toolbox, text=component, font=("Arial", 10), width=20, height=2,
                        bg="#3498db", fg="white", command=lambda c=component: self.select_component(c))
            btn.pack(pady=5, padx=10)
    
    def create_workspace(self):
        self.workspace = Frame(self.root, bg="white", relief="sunken", bd=2)
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        title_label = Label(self.workspace, text="Workspace", font=("Arial", 16, "bold"), bg="white")
        title_label.pack(pady=10)
        
        self.component_label = Label(self.workspace, text="Selected Component: None",
                                   font=("Arial", 12), bg="white")
        self.component_label.pack(pady=5)
        
        self.file_label = Label(self.workspace, text="File Selected: None", 
                               font=("Arial", 12), bg="white")
        self.file_label.pack(pady=5)
        
        self.input_btn = Button(self.workspace, text="Input Data", font=("Arial", 12), 
                               bg="#27ae60", fg="white", command=self.select_file, width=15, height=2)
        self.input_btn.pack(pady=20)
        
        self.execute_btn = Button(self.workspace, text="Execute", font=("Arial", 14, "bold"), 
                                 bg="#e74c3c", fg="white", command=self.execute, width=15, height=2)
        self.execute_btn.pack(pady=10)
        
        self.results_frame = Frame(self.workspace, bg="white")
        self.results_frame.pack(pady=20, fill="both", expand=True)
    
    def create_insight_panel(self):
        self.insight = Frame(self.root, bg="#ecf0f1")
        self.insight.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        title_label = Label(self.insight, text="AI Insights", font=("Arial", 14, "bold"), bg="#ecf0f1")
        title_label.pack(pady=10)
        
        self.code_btn = Button(self.insight, text="Generate Code", font=("Arial", 10), 
                              bg="#9b59b6", fg="white", command=self.generate_code, width=20)
        self.code_btn.pack(pady=5)
        
        self.explain_btn = Button(self.insight, text="Generate Explanation", font=("Arial", 10), 
                                 bg="#8e44ad", fg="white", command=self.generate_explanation, width=20)
        self.explain_btn.pack(pady=5)
        
        self.insight_text = Text(self.insight, wrap="word", font=("Arial", 10), bg="white", height=30)
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
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            data = data.dropna()
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            self.feature_columns = X.columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.trained_model = LinearRegression()
            self.trained_model.fit(X_train_scaled, y_train)
            
            predictions = self.trained_model.predict(X_test_scaled)
            r2 = r2_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Linear Regression Model Trained!\nR² Score: {r2:.4f}")
            self.create_prediction_interface("regression")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def logistic_regression(self):
        try:
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            data = data.dropna()
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            self.feature_columns = X.columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.trained_model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
            self.trained_model.fit(X_train_scaled, y_train)
            
            predictions = self.trained_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Logistic Regression Model Trained!\nAccuracy: {accuracy:.4f}")
            self.create_prediction_interface("classification")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def text_classification(self):
        try:
            if self.selected_file.endswith('.csv'):
                data = pd.read_csv(self.selected_file)
            else:
                data = pd.read_excel(self.selected_file)
            
            X = data.iloc[:, 0].astype(str)
            y = data.iloc[:, 1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.trained_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            self.trained_model.fit(X_train, y_train)
            predictions = self.trained_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            messagebox.showinfo("Model Info", f"Text Classification Model Trained!\nAccuracy: {accuracy:.4f}")
            self.create_text_prediction_interface()
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def image_classification(self):
        messagebox.showinfo("Image Classification", 
                          "Image Classification requires pre-trained models.\nPlease upload a model file.")
    
    def create_prediction_interface(self, model_type):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        pred_label = Label(self.results_frame, text="Make Predictions", 
                          font=("Arial", 14, "bold"), bg="white")
        pred_label.pack(pady=10)
        
        self.input_vars = {}
        for col in self.feature_columns:
            frame = Frame(self.results_frame, bg="white")
            frame.pack(pady=2, fill="x", padx=20)
            Label(frame, text=f"{col}:", font=("Arial", 10), bg="white").pack(side="left")
            var = tk.StringVar()
            tk.Entry(frame, textvariable=var, font=("Arial", 10)).pack(side="right", padx=10)
            self.input_vars[col] = var
        
        Button(self.results_frame, text="Predict", font=("Arial", 12), bg="#f39c12", fg="white",
               command=lambda: self.make_prediction(model_type)).pack(pady=20)
        
        self.prediction_result = Label(self.results_frame, text="", font=("Arial", 12, "bold"), bg="white")
        self.prediction_result.pack(pady=10)
    
    def create_text_prediction_interface(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        Label(self.results_frame, text="Text Classification", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        Label(self.results_frame, text="Enter text:", font=("Arial", 10), bg="white").pack(pady=5)
        
        self.text_input = Text(self.results_frame, height=5, width=50)
        self.text_input.pack(pady=10)
        
        Button(self.results_frame, text="Classify", font=("Arial", 12), bg="#f39c12", fg="white",
               command=self.classify_text).pack(pady=10)
        
        self.text_result = Label(self.results_frame, text="", font=("Arial", 12, "bold"), bg="white")
        self.text_result.pack(pady=10)
    
    def make_prediction(self, model_type):
        try:
            input_data = []
            for col in self.feature_columns:
                value = self.input_vars[col].get()
                if not value:
                    messagebox.showwarning("Warning", f"Please enter value for {col}")
                    return
                input_data.append(float(value))
            
            input_scaled = self.scaler.transform([input_data])
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
                messagebox.showwarning("Warning", "Please enter text")
                return
            
            prediction = self.trained_model.predict([text])[0]
            self.text_result.config(text=f"Classification: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def generate_code(self):
        if not self.groq_client:
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, "Groq API not configured. Install groq package.")
            return
        
        if not self.selected_component:
            messagebox.showwarning("Warning", "Select a component first!")
            return
        
        try:
            prompt = f"Generate Python code for {self.selected_component} using scikit-learn with comments."
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, response.choices[0].message.content)
        except Exception as e:
            messagebox.showerror("Error", f"Code generation failed: {str(e)}")
    
    def generate_explanation(self):
        if not self.groq_client:
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, "Groq API not configured. Install groq package.")
            return
        
        if not self.selected_component:
            messagebox.showwarning("Warning", "Select a component first!")
            return
        
        try:
            prompt = f"Explain {self.selected_component} in simple terms with examples."
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            self.insight_text.delete("1.0", tk.END)
            self.insight_text.insert(tk.END, response.choices[0].message.content)
        except Exception as e:
            messagebox.showerror("Error", f"Explanation failed: {str(e)}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AIFlowPlatform()
    app.run()
