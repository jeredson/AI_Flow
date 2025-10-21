from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

groq_client = Groq(api_key="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")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        model_type = request.form['model_type']
        file = request.files['dataset']
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)
        
        data = data.dropna()
        
        if model_type == 'text':
            X = data.iloc[:, 0].astype(str)
            y = data.iloc[:, 1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            metric = 'Accuracy'
        else:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            if model_type == 'logistic' and y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if model_type == 'linear':
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                score = r2_score(y_test, predictions)
                metric = 'R² Score'
            else:
                model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                score = accuracy_score(y_test, predictions)
                metric = 'Accuracy'
            
            session['scaler'] = pickle.dumps(scaler)
            session['features'] = X.columns.tolist()
        
        session['model'] = pickle.dumps(model)
        session['model_type'] = model_type
        
        return jsonify({'success': True, 'metric': metric, 'score': f'{score:.4f}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = pickle.loads(session['model'])
        model_type = session['model_type']
        
        if model_type == 'text':
            text = request.form['text']
            prediction = model.predict([text])[0]
            return jsonify({'success': True, 'prediction': str(prediction)})
        else:
            scaler = pickle.loads(session['scaler'])
            features = session['features']
            input_data = [float(request.form[f]) for f in features]
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]
            return jsonify({'success': True, 'prediction': f'{prediction:.4f}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        action = request.form['action']
        component = request.form['component']
        
        if action == 'code':
            prompt = f"Generate Python code for {component} using scikit-learn with comments."
        else:
            prompt = f"Explain {component} in simple terms with examples."
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        
        return jsonify({'success': True, 'content': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
