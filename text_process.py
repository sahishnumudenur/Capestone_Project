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
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Remove short words (optional)
    tokens = [word for word in tokens if len(word) > 2]

    # 10. Join tokens back to string
    cleaned_text = " ".join(tokens)

    return cleaned_text

case_text='In Maya Infotech Pvt. Ltd. vs. Neel Sharma (2023), an employee was accused of unauthorized access to the company’s internal database and leaking client information to a competitor. Digital forensics revealed log-in attempts outside office hours, IP address traces, and copies of confidential files on the accused’s personal device. The court held that the accused violated the IT Act, 2000, particularly Sections 43 and 66 concerning unauthorized access and data theft. The defendant was ordered to pay damages to the company and faced two years of imprisonment. This case set a precedent for strict action against insider cyber threats in corporate environments.'

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