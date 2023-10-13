"""
Project Name: Enhanced Email Relationship and Context Analyzer
Last Modified: 2023-09-19  # [Current Date]
Author: Josh Klein & ChatGPT (OpenAI)
Version: 5.3

Description:
The script processes .eml files to analyze relationships and context within email communications. 
It uses NLP (Natural Language Processing) techniques, including tokenization, sentiment analysis, 
named entity recognition, stemming, and topic modeling (via Sklearn's LDA). The processed data is 
exported to a JSON file, preserving the complex data structure for further analysis and insights.

Goals:
1. Extract and process relevant data and context from .eml files, including handling multipart email bodies.
2. Identify entities and topics of discussion from the email communications.
3. Provide a sentiment score for the communication.
4. Implement relationship scoring to gauge interaction intensity.
5. Provide meaningful insights into email communications and relationships that could be used for networking, 
   relationship management, and more.

Possible new features to add:
1. Relationship Scoring: Relationship scoring is currently a simple count of named entities recognized in the text. Depending on the specific use-case, you might want to enhance and sophisticate this functionality. For example, if an entity appears in multiple emails, this could have a different weight compared to multiple mentions within a single email.
2. Cleaning and Preprocessing: Ensure that data cleaning (e.g., removing special characters, handling various encodings, etc.) is sufficiently robust to handle real-world messy data. This might include further HTML cleaning in extract_body or further text preprocessing in nlp_processing.
3. Metadata Processing: Currently, it returns email address domains and local parts as separate entities which might not be ideal for all use-cases. Consider concatenating them back into a full email address or handling them in a way that best suits your use-case.
4. Documentation: Ensure thorough documentation for every function and segment to ensure maintainability and facilitate any future development or debugging.
5. Configurability: Consider moving configurable parameters (like n_components in LDA) to a configuration file or command-line arguments to enhance usability and configurability without modifying the code.

References:
1. NLTK for tokenization, named entity recognition, sentiment analysis, and stemming.
2. BeautifulSoup (bs4) for parsing HTML content from emails.
3. Sklearn for topic modeling using Latent Dirichlet Allocation (LDA).
4. Multiprocessing and tqdm for efficient and parallelized processing with a progress bar.

Usage Notes:
- Ensure nltk data is downloaded (e.g., nltk.download() for stopwords, punkt, etc.)
- Install necessary libraries (nltk, bs4, sklearn, tqdm) via pip (e.g., pip install nltk bs4 sklearn tqdm).

Modules Used:
- os: For file and directory operations
- json: For exporting data in JSON format
- logging: To keep logs of process and errors
- nltk: For various NLP tasks
- email: To parse .eml files and extract data
- multiprocessing: To utilize multiple cores for parallel processing
- tqdm: To display a progress bar
- sklearn: For topic modeling using Latent Dirichlet Allocation
- bs4 (BeautifulSoup): To parse HTML content in emails
- collections (defaultdict): To maintain relationship scores

Version History:
- Version 6 Moved most recent version of the code to Visual Studio Code
- Installed Git, CoPilot, and Vim plugins to Visual Studio Code to run a 30 trial of CoPilot
"""

import os
import json
import logging
import nltk
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from collections import defaultdict
from bs4 import BeautifulSoup
import os
import json
import logging
import nltk
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
from bs4 import BeautifulSoup
from threading import Lock
scores_lock = Lock()

logging.basicConfig(filename='email_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sia = SentimentIntensityAnalyzer()
porter = PorterStemmer()
relationship_scores = defaultdict(int)
stop_words = set(stopwords.words('english'))

email_files = []
script_directory = os.path.dirname(os.path.abspath(__file__))
for root, dirs, files in os.walk(script_directory):
    for file in files:
        if file.endswith('.eml'):
            email_files.append(os.path.join(root, file))

def parse_eml(eml_file_path):
    try:
        with open(eml_file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject, from_, to_, date = extract_email_metadata(msg, eml_file_path)
        email_body = extract_body(msg, eml_file_path)

        if not email_body:  # Additional check to ensure not processing None objects.
            logging.warning(f"Email body empty or None in file {eml_file_path}. Skipping processing.")
            return None

        if not validate_email_body(email_body):
            logging.error(f"Invalid email body in file {eml_file_path}. Skipping processing.")
            return None

        tokens, sentiment, named_entities, stemmed_tokens, keywords = nlp_processing(email_body)
        for entity in named_entities:
            if isinstance(entity, tuple):
                continue
            name = ' '.join(e[0] for e in entity)
            # relationship_scores[name] += 1        depreciated to allow fo concurrency controls
            update_relationship_scores(name, 1)
        topics = topic_modeling(tokens)

        return {
            'subject': subject,
            'from': from_,
            'to': to_,
            'date': date,
            'email_body': email_body,
            'sentiment': sentiment['compound'],
            'named_entities': str(named_entities),
            'stemmed_tokens': stemmed_tokens,
            'keywords': keywords,
            'topics': topics.tolist() if topics is not None else [],
            'relationship_scores': dict(relationship_scores)
        }
    except Exception as e:
        logging.error(f"Error parsing {eml_file_path}: {e}")
        return None

def extract_email_metadata(msg, eml_file_path):
    subject = msg.get('subject', '')
    from_ = getaddresses(msg.get_all('from', []))
    to_ = getaddresses(msg.get_all('to', []))
    date = parsedate_to_datetime(msg['date']) if msg['date'] else None

    return subject, from_, to_, date.isoformat() if date else None

def extract_body(msg, eml_file_path):
    if msg.is_multipart():
        for part in msg.iter_parts():
            content_type = part.get_content_type()
            charset = part.get_content_charset() if part.get_content_charset() else 'utf-8'
            if content_type == 'text/plain':
                return part.get_payload(decode=True).decode(charset, errors='ignore')
            elif content_type == 'text/html':
                html_content = part.get_payload(decode=True).decode(charset, errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text()
    else:
        charset = msg.get_content_charset() if msg.get_content_charset() else 'utf-8'
        return msg.get_payload(decode=True).decode(charset, errors='ignore')

def nlp_processing(email_body):
    # Create a CountVectorizer object with the desired parameters
    vectorizer = CountVectorizer(stop_words='english', min_df=0.01, max_df=0.95)

    # Use the vectorizer to transform the email body text into a document-term matrix
    X = vectorizer.fit_transform([email_body])

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names()

    # Convert the document-term matrix to a pandas DataFrame
    df = pd.DataFrame(X.toarray(), columns=feature_names)

    # Get the tokens, sentiment, named entities, stemmed tokens, and keywords
    tokens = word_tokenize(email_body)
    sentiment = sia.polarity_scores(email_body)
    named_entities = ne_chunk(pos_tag(tokens))
    stemmed_tokens = [porter.stem(t) for t in tokens]
    keywords = [t for t in tokens if t.lower() not in stop_words]

    # Return the DataFrame, tokens, sentiment, named entities, stemmed tokens, and keywords
    return df, tokens, sentiment, named_entities, stemmed_tokens, keywords

def validate_email_body(email_body: str) -> bool:
    """
    Validate the email body content before processing.
    """
    return bool(email_body) and isinstance(email_body, str)

def topic_modeling(tokens):
    text_data = [' '.join(tokens)]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(text_data)
    
    nmf = NMF(n_components=5, max_iter=100, random_state=0).fit(tfidf)
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = None
    
    if nmf.components_.any():
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            topics.append([tfidf_feature_names[i] for i in topic.argsort()[:-5 - 1:-1]])
    return topics

def parallel_processing(email_files):
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_eml, email_files), total=len(email_files)))
    return [result for result in results if result is not None]

def export_to_json(parsed_data, filename="processed_emails.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data exported to {filename}")

def update_relationship_scores(name: str, score: int) -> None:
    """
    Safely update the relationship_scores dictionary in a multi-threading environment.
    
    Parameters:
        name (str): The key in the relationship_scores to update.
        score (int): The value to add to the existing score for 'name'.
    """
    global relationship_scores
    with scores_lock:  # Ensure thread safety with a lock
        # Update the score in a thread-safe manner
        if name in relationship_scores:
            relationship_scores[name] += score
        else:
            relationship_scores[name] = score

if __name__ == "__main__":
    # Ensure that required NLTK data is downloaded (like stopwords, vader lexicon etc.)
    nltk.download(['stopwords', 'vader_lexicon', 'punkt', 'maxent_ne_chunker', 'words'])

    # Process emails in parallel
    parsed_emails = parallel_processing(email_files)

    # Export processed data to JSON
    export_to_json(parsed_emails)