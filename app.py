from flask import Flask, render_template, request
from typing import Optional, Dict
import platform
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from flask import jsonify
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.tokenize import RegexpTokenizer
from urllib.parse import urlparse
import tldextract
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Custom transformer class
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

# Parse URL function
def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "scheme": None,  # not established a value for this
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except:
        return None

# Other functions, classes, and code snippets from the original code

# Load dataset
df = pd.read_csv(r'C:\Users\H P\OneDrive\Desktop\fnl\dataset_phishing.csv')

# Group by URL and preprocess
df_grp = df.groupby(["url"])[["status"]].sum().reset_index()

def get_num_subdomains(netloc: str) -> int:
    subdomain = tldextract.extract(netloc).subdomain 
    if subdomain == "":
        return 0
    return subdomain.count('.') + 1

# Feature engineering steps
df_grp["parsed_url"] = df_grp.url.apply(parse_url)
df_grp = pd.concat([
    df_grp.drop(['parsed_url'], axis=1),
    df_grp['parsed_url'].apply(pd.Series)
], axis=1)
df_grp = df_grp[~df_grp.netloc.isnull()]
df_grp["length"] = df_grp.url.str.len()
df_grp["tld"] = df_grp.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
df_grp['tld'] = df_grp['tld'].replace('', 'None')
df_grp["is_ip"] = df_grp.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
df_grp['domain_hyphens'] = df_grp.netloc.str.count('-')
df_grp['domain_underscores'] = df_grp.netloc.str.count('_')
df_grp['path_hyphens'] = df_grp.path.str.count('-')
df_grp['path_underscores'] = df_grp.path.str.count('_')
df_grp['slashes'] = df_grp.path.str.count('/')
df_grp['full_stops'] = df_grp.path.str.count('.')

df_grp['num_subdomains'] = df_grp['netloc'].apply(lambda net: get_num_subdomains(net))

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
def tokenize_domain(netloc: str) -> str:
    split_domain = tldextract.extract(netloc)
    no_tld = str(split_domain.subdomain + '.' + split_domain.domain)
    return " ".join(map(str, tokenizer.tokenize(no_tld)))

df_grp['domain_tokens'] = df_grp['netloc'].apply(lambda net: tokenize_domain(net))

df_grp['path_tokens'] = df_grp['path'].apply(lambda path: " ".join(map(str, tokenizer.tokenize(path))))

df_grp_y = df_grp['status']
df_grp.drop('status', axis=1, inplace=True)
df_grp.drop('url', axis=1, inplace=True)
df_grp.drop('scheme', axis=1, inplace=True)
df_grp.drop('netloc', axis=1, inplace=True)
df_grp.drop('path', axis=1, inplace=True)
df_grp.drop('params', axis=1, inplace=True)
df_grp.drop('query', axis=1, inplace=True)
df_grp.drop('fragment', axis=1, inplace=True)

# Machine learning pipeline
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(df_grp, df_grp_y, test_size=0.2)

numeric_features = ['length', 'domain_hyphens', 'domain_underscores', 'path_hyphens', 'path_underscores', 'slashes', 'full_stops', 'num_subdomains']
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

categorical_features = ['tld', 'is_ip']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

vectorizer_features = ['domain_tokens', 'path_tokens']
vectorizer_transformer = Pipeline(steps=[
    ('con', Converter()),
    ('tf', TfidfVectorizer())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('domvec', vectorizer_transformer, ['domain_tokens']),
        ('pathvec', vectorizer_transformer, ['path_tokens'])
    ])

svc_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearSVC())])

log_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

nb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', MultinomialNB())])

svc_clf.fit(X_train, y_train)
log_clf.fit(X_train, y_train)
nb_clf.fit(X_train, y_train)

def get_system_info():
    # Example: Get information about the operating system
    try:
        os_type = platform.system()
        authentication = os.getlogin() if os.name == 'posix' else None
        firewall_protection = "Automatically Obtained"  # You may need additional logic to determine firewall status
        antivirus_status = "Automatically Obtained"  # You may need additional logic to determine antivirus status
        data_encryption = "Automatically Obtained"  # You may need additional logic to determine data encryption status
    except Exception as e:
        print(f"Error retrieving system information: {e}")
        os_type = authentication = firewall_protection = antivirus_status = data_encryption = "Unknown"

    return {
        'os_type': os_type,
        'authentication': authentication,
        'firewall_protection': firewall_protection,
        'antivirus_status': antivirus_status,
        'data_encryption': data_encryption
    }

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
         # Extract form data
        url_input = request.form.get('url_input')

        # Automatically retrieve system information
        system_info = get_system_info()

        # Extract relevant information
        os_type = system_info['os_type']
        authentication = system_info['authentication']
        firewall_protection = system_info['firewall_protection']
        antivirus_status = system_info['antivirus_status']
        data_encryption = system_info['data_encryption']
        
        # Perform the necessary transformations on the input data
        parsed_url = parse_url(url_input)


        # Pass the system information to the result.html template
        prediction = {
            'os_type': os_type,
            'authentication': authentication,
            'firewall_protection': firewall_protection,
            'antivirus_status': antivirus_status,
            'data_encryption': data_encryption
        }

        # Create a DataFrame similar to your training data for the user input
        input_data = pd.DataFrame({
            'length': len(url_input),
            'tld': tldextract.extract(parsed_url['netloc']).suffix,
            'is_ip': parsed_url['netloc'].isdigit(),
            'domain_hyphens': parsed_url['netloc'].count('-'),
            'domain_underscores': parsed_url['netloc'].count('_'),
            'path_hyphens': parsed_url['path'].count('-'),
            'path_underscores': parsed_url['path'].count('_'),
            'slashes': parsed_url['path'].count('/'),
            'full_stops': parsed_url['path'].count('.'),
            'num_subdomains': get_num_subdomains(parsed_url['netloc']),
            'domain_tokens': tokenize_domain(parsed_url['netloc']),
            'path_tokens': " ".join(map(str, tokenizer.tokenize(parsed_url['path']))),
            'authentication': authentication,
            'os_type': os_type,
            'firewall_protection': firewall_protection,
            'antivirus_status': antivirus_status,
            'data_encryption': data_encryption
        }, index=[0])

        # Use the trained model to make predictions
        prediction = svc_clf.predict(input_data)

        return render_template('result.html', prediction=prediction)

# Evaluation function
def results(name: str, model: BaseEstimator) -> None:
    preds = model.predict(X_test)

    print(name + " score: %.3f" % model.score(X_test, y_test))
    print(classification_report(y_test, preds))
    labels = ['Good', 'Bad']

    conf_matrix = confusion_matrix(y_test, preds)

    font = {'family': 'sans-serif'}
    plt.rc('font', **font)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Greens')
    plt.title("Confusion Matrix for " + name)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
