from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import torch

# Bag of Words + Rule-based Scoring
class BagOfWordsRater:
    def __init__(self):
        # Basic positive word list
        self.pos_words = {
            'good', 'nice', 'beautiful', 'pretty', 'lovely', 'sweet', 'cute', 
            'happy', 'love', 'like', 'best', 'perfect', 'amazing', 'wonderful',
        }
        
        # Basic negative word list
        self.neg_words = {
            'bad', 'ugly', 'hate', 'worst', 'boring', 'tired', 'sick',
            'never', 'no', 'not', 'cant', 'wont', 'dont', 'cause'
        }
        
        # Basic structural rules
        self.rules = [
            (lambda line: '?' in line, 0.2),                     # Has question
            (lambda line: len(line.split()) > 15, -0.2),         # Too long
            (lambda line: len(line.split()) < 3, -0.3),          # Too short
            (lambda line: '!' in line, 0.1),                     # Shows enthusiasm
            (lambda line: 'you' in line.lower(), 0.1),           # Personal reference
            (lambda line: 'because' in line.lower(), 0.1),       # Has explanation
        ]

    def rate_pickup_line(self, line):
        words = line.lower().split()
        
        # Count positive and negative words
        pos_count = sum(1 for word in words if word in self.pos_words)
        neg_count = sum(1 for word in words if word in self.neg_words)
        
        # Calculate base sentiment score
        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words == 0:
            base_score = 0.2  # Neutral by default
        else:
            base_score = pos_count / (pos_count + neg_count)
        
        # Apply simple rules
        rule_score = sum(score for rule, score in self.rules if rule(line))
        
        # Combine scores
        final_score = (base_score * 0.7) + (rule_score * 0.3)
        
        # Normalize score to [0, 1]
        return max(0.0, min(1.0, final_score))

# VADER Lexicon Scoring
class VADERLexiconRater:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def rate_pickup_line(self, line):
        scores = self.analyzer.polarity_scores(line)
        
        # Get component scores
        compound = (scores['compound'] + 1) / 2  # Normalize compound from [-1,1] to [0,1]
        pos = scores['pos']      # Already [0,1]
        neg = scores['neg']      # Already [0,1]
        neu = scores['neu']      # Already [0,1]

        weighted_score = (
            compound * 0.4 +     # Overall sentiment
            pos * 0.4 +          # Positive sentiment is important
            (1 - neg) * 0.1 +    # Penalize negative sentiment
            (1 - neu) * 0.1      # Slight penalty for too much neutrality
        )
        
        # Normalize score to [0, 1]
        return max(0.0, min(1.0, weighted_score))

# Naive Bayes Classifier
class NaiveBayesRater:
    def __init__(self, train_data, train_labels):
        # Initialzing TF-IDF Vectorizer and Multinomial Naive Bayes Classifier
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Use bigrams
            max_features=5000,
            min_df=2,
            stop_words='english',
            lowercase=True,
            strip_accents="unicode"
        )
        self.classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        
        # Fit the pipeline
        X = self.vectorizer.fit_transform(train_data)
        self.classifier.fit(X, train_labels)

    def rate_pickup_line(self, line):
        X = self.vectorizer.transform([line])
        prob = self.classifier.predict_proba(X)[0][1]

        # Returning normalized probability
        return prob


# Markov Chain Scoring
class MarkovChainRater:
    def __init__(self, train_data, train_labels):
        self.flirty_transitions = defaultdict(lambda: defaultdict(float))
        self.nonflirty_transitions = defaultdict(lambda: defaultdict(float))
        self.flirty_counts = defaultdict(float)
        self.nonflirty_counts = defaultdict(float)
        self.train(train_data, train_labels)

    def preprocess(self, text):
        # Simple preprocessing to preserve patterns
        return text.lower().strip()

    def train(self, texts, labels):
        # Train separate models for flirty and non-flirty texts
        for text, label in zip(texts, labels):
            words = self.preprocess(text).split()
            
            # Add start/end tokens
            words = ['<START>'] + words + ['<END>']
            
            # Count transitions
            for i in range(len(words) - 1):
                if label == 1:  # Flirty
                    self.flirty_transitions[words[i]][words[i+1]] += 1
                    self.flirty_counts[words[i]] += 1
                else:  # Non-flirty
                    self.nonflirty_transitions[words[i]][words[i+1]] += 1
                    self.nonflirty_counts[words[i]] += 1

        # Add smoothing
        smooth = 1.0
        vocab = set()
        for d in [self.flirty_transitions, self.nonflirty_transitions]:
            for w1 in d:
                vocab.update(d[w1].keys())
                vocab.add(w1)
        
        # Apply smoothing to all word pairs
        for w1 in vocab:
            for w2 in vocab:
                self.flirty_transitions[w1][w2] += smooth
                self.nonflirty_transitions[w1][w2] += smooth
                self.flirty_counts[w1] += smooth
                self.nonflirty_counts[w1] += smooth

    def get_sequence_probability(self, words, transitions, counts):
        # Calculate probability of sequence using log probabilities
        log_prob = 0
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            prob = transitions[w1][w2] / counts[w1] if counts[w1] > 0 else 0
            log_prob += np.log(prob + 1e-10)  # Add small constant to avoid log(0)
        return log_prob

    def rate_pickup_line(self, text):
        words = self.preprocess(text).split()
        words = ['<START>'] + words + ['<END>']
        
        # Get probabilities under both models
        flirty_prob = self.get_sequence_probability(words, self.flirty_transitions, self.flirty_counts)
        nonflirty_prob = self.get_sequence_probability(words, self.nonflirty_transitions, self.nonflirty_counts)
        
        # Convert log probabilities to score
        score = np.exp(flirty_prob) / (np.exp(flirty_prob) + np.exp(nonflirty_prob))
        
        return score


class BERTRater:
    def __init__(self):
        # Use a model fine-tuned for sentiment/emotion detection
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
    def rate_pickup_line(self, line):
        try:
            # Tokenize input
            inputs = self.tokenizer(
                line,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities for positive, neutral, negative
                probabilities = torch.softmax(logits, dim=1)
                
                # Use positive sentiment as a proxy for flirtiness
                score = probabilities[0][2].item()  # Index 2 is positive class
                
            return score

        except Exception as e:
            print(f"Error processing line: {e}")
        

# GPT-based Sentiment Scoring
class GPTRater:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.sentiment_classifier = pipeline("sentiment-analysis")
        self.min_length = 3

    def rate_pickup_line(self, line):
        if len(line.split()) < self.min_length:
            return 0.1

        # Check multiple aspects of the line
        candidate_labels = [
            "clever romantic pickup line",
            "flirty conversation starter",
            "genuine compliment",
            "cheesy pickup line",
            "generic greeting",
            "unrelated statement",
            "not funny"
        ]

        # Get zero-shot classification results
        result = self.classifier(line, candidate_labels)
        
        # Get sentiment score
        sentiment = self.sentiment_classifier(line)[0]
        sentiment_score = 1.0 if sentiment['label'] == 'POSITIVE' else 0.2
        
        # Calculate base score from positive categories
        base_score = 0.0
        for label, score in zip(result['labels'], result['scores']):
            if "pickup" in label or "flirty" in label or "compliment" in label:
                weight = 1.0 if "clever" in label or "genuine" in label else 0.7
                base_score += score * weight
            elif "unrelated" in label:
                base_score -= score * 0.8
            elif "not funny" in label:
                base_score -= score * 0.7

        # Apply modifiers
        length_bonus = min(1.0, len(line.split()) / 12.0)  # Optimal length around 8-12 words
        creativity_score = 1.0 - result['scores'][-1]  # Penalty for being unrelated
        
        # Combine all factors
        final_score = (
            base_score * 0.6 +           # Base classification
            sentiment_score * 0.3 +      # Sentiment impact
            length_bonus * 0.1 +         # Length contribution
            creativity_score * 0.1       # Creativity factor
        )

        return max(0.0, min(1.0, final_score))  # Ensure score is between 0 and 1


# Load Dataset
def load_data():
    dataset = load_dataset("ieuniversity/flirty_or_not")
    train_data = dataset['train']['texts']
    train_labels = dataset['train']['label']
    val_data = dataset['validation']['texts']
    val_labels = dataset['validation']['label']
    test_data = dataset['test']['texts']
    test_labels = dataset['test']['label']
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# Evaluate All Systems
def evaluate_systems(pickup_lines, train_data, train_labels):
    bow_rater = BagOfWordsRater()
    vader_rater = VADERLexiconRater()
    nb_rater = NaiveBayesRater(train_data, train_labels)
    mc_rater = MarkovChainRater(train_data, train_labels)
    bert_rater = BERTRater()
    gpt_rater = GPTRater()
    
    results = []
    for line in pickup_lines:
        bow_rating = bow_rater.rate_pickup_line(line)
        vader_rating = vader_rater.rate_pickup_line(line)
        nb_rating = nb_rater.rate_pickup_line(line)
        mc_rating = mc_rater.rate_pickup_line(line)
        bert_rating = bert_rater.rate_pickup_line(line)
        gpt_rating = gpt_rater.rate_pickup_line(line)
        
        results.append((line, bow_rating, vader_rating, nb_rating, mc_rating, bert_rating, gpt_rating))
    
    return results


# Main Script
if __name__ == "__main__":
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()
    
    pickup_lines = [
        "You are amazing and beautiful!",
        "Did it hurt when you fell from heaven?",
        "Are you a magician? Because whenever I look at you, everyone else disappears.",
        "Do you have a map? I keep getting lost in your eyes.",
        "Is your name Google? Because you have everything I’ve been searching for.",
        "You must be tired because you’ve been running through my mind all day.",
        "Are you a parking ticket? Because you’ve got FINE written all over you.",
        "Do you believe in love at first sight, or should I walk by again?",
        "I must be a snowflake, because I’ve fallen for you.",
        "If you were a vegetable, you’d be a cute-cumber."
    ]
    
    results = evaluate_systems(pickup_lines, train_data, train_labels)
    
    for line, normalized_bow_rating, normalized_vader_rating, normalized_nb_rating, normalized_mc_rating, normalized_bert_rating, normalized_gpt_rating in results:
        print(f"Pickup Line: {line}")
        print(f"Normalized Bag of Words Rating: {normalized_bow_rating:.2f}")
        print(f"Normalized VADER Rating: {normalized_vader_rating:.2f}")
        print(f"Normalized Naive Bayes Rating: {normalized_nb_rating:.2f}")
        print(f"Normalized Markov Chain Rating: {normalized_mc_rating:.2f}")
        print(f"Normalized BERT Rating: {normalized_bert_rating:.2f}")
        print(f"Normalized GPT Rating: {normalized_gpt_rating:.2f}")
        print()
