from flask import Flask, render_template, request, redirect, session, url_for, flash
import mysql.connector
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)
app.secret_key = "your_secret_key_here"
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 5. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 6. Tokenize
    tokens = word_tokenize(text)

    # 7. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Lemmatization
    #tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Remove short words (optional)
    tokens = [word for word in tokens if len(word) > 2]

    # 10. Join tokens back to string
    cleaned_text = " ".join(tokens)

    return cleaned_text

# -----------------------------------------------------------
# 1. CREATE DATABASE & TABLES
# -----------------------------------------------------------
def init_database():
    root_conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
charset="utf8",
    )
    cursor = root_conn.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS case_system")
    root_conn.commit()
    root_conn.close()

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(150) UNIQUE,
        password VARCHAR(255)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        case_text TEXT,
                   label_pred TEXT,
                   category_pred TEXT,
                   proof_sentence TEXT
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


@app.route('/')
def home():
    return redirect('/login')

# -----------------------------------------------------------
# 2. DATABASE CONNECTION
# -----------------------------------------------------------
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="case_system",
        charset="utf8",
    )


# -----------------------------------------------------------
# 3. SIGNUP (plain password)
# -----------------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']  # <-- plain text

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            flash("Email already exists!", "danger")
            return redirect('/signup')

        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                       (name, email, password))
        conn.commit()

        flash("Signup successful! Login now.", "success")
        return redirect('/login')

    return render_template('signup.html')


# -----------------------------------------------------------
# 4. LOGIN (plain password check)
# -----------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_input = request.form['password']

        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s",
                       (email, password_input))
        user = cursor.fetchone()

        if user:
            session['user_id'] = user['id']
            session['name'] = user['name']
            return redirect('/dashboard')

        flash("Invalid login!", "danger")
        return redirect('/login')

    return render_template('login.html')


# -----------------------------------------------------------
# 5. DASHBOARD
# -----------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    return render_template('dashboard.html', name=session['name'])

import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
df = pd.read_csv("case_files_total.csv")

# Load everything
label_model = joblib.load("legal_model.pkl")
category_model = joblib.load("best_model1.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("Models loaded successfully!")

# -----------------------------------------------------------
# 6. SUBMIT CASE
# -----------------------------------------------------------
@app.route('/submit_case', methods=['GET', 'POST'])
def submit_case():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        case_text = request.form['case_text']

        conn = get_db()
        cursor = conn.cursor()

        
        processed = preprocess_text(case_text)
        print(processed)


        stemmer = PorterStemmer()

        def stem_tokens(tokens):
            return [stemmer.stem(word) for word in tokens]



        # Tokenize processed text
        tokens = word_tokenize(processed)

        # Apply POS tagging
        pos_tags = pos_tag(tokens)

        print(pos_tags)
        new_case = [case_text]
        
        new_case_vec = vectorizer.transform(new_case)
        # Predict label
        label_pred = label_model.predict(new_case_vec)[0]
        print(label_pred)
        if label_pred==0:
            label_pred='Accepted'
        elif label_pred==1:
            label_pred='Other'
        else:
            label_pred='Rejected'
        # Vectorize
        new_vec = vectorizer.transform(new_case)
        # Predict category
        category_pred = category_model.predict(new_vec)[0]
        print(category_pred)
        # Extract proof sentence using similarity
        all_cases_vec = vectorizer.transform(df['case_info'].astype(str))
        
        similarities = cosine_similarity(new_vec, all_cases_vec)
        print(similarities)
        best_match_index = similarities.argmax()
        print(best_match_index)
        proof_sentence = df.iloc[best_match_index]['proof_sentence']
        print(proof_sentence)
        # Output
        print("Predicted Verdict:", label_pred)
        print("Predicted Category:", category_pred)
        print("Proof Sentence:", proof_sentence)
        print(session['user_id'])
        print('___________________')
        print(case_text)
        print('___________________')
        print(label_pred,type(label_pred))
        print('___________________')
        print(category_pred)
        print('___________________')
        print(proof_sentence)
        print('___________________')
        cursor.execute("INSERT INTO cases (user_id, case_text,label_pred,category_pred,proof_sentence) VALUES (%s, %s,%s, %s,%s)",
                       (session['user_id'], case_text,label_pred,category_pred,proof_sentence))
        conn.commit()
        
        pred=''
        if label_pred==0:
            pred='Accepted'
        elif label_pred==1:
            pred='Other'
        else:
            pred='Rejected'

        flash("Case submitted!", "Predicted Result "+pred)
        f=open('output.txt','w')
        f.write("Predicted Result "+pred)
        f.close()
        f=open('output1.txt','w')
        f.write("Predicted Case category "+category_pred)
        f.close()
        f=open('output2.txt','w')
        f.write("Proof Sentence "+proof_sentence)
        f.close()
        return redirect('/my_cases')

    return render_template('submit_case.html')


# -----------------------------------------------------------
# 7. MY CASES
# -----------------------------------------------------------
@app.route('/my_cases')
def my_cases():
    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM cases WHERE user_id=%s ORDER BY id DESC",
                   (session['user_id'],))
    cases = cursor.fetchall()
    f=open('output.txt','r')
    pred=f.read()
    f.close()
    f=open('output1.txt','r')
    category_pred=f.read()
    f.close()
    f=open('output2.txt','r')
    proof_sentence=f.read()
    f.close()

    return render_template('my_cases.html', cases=cases,pred=pred,category_pred=category_pred,proof_sentence=proof_sentence)


# -----------------------------------------------------------
# 8. LOGOUT
# -----------------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# -----------------------------------------------------------
# RUN APP
# -----------------------------------------------------------
if __name__ == "__main__":
    #init_database()
    app.run(debug=True)
