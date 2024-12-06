from flask import Flask, request, jsonify
from flask_cors import CORS

# Import or mock the rating systems
from systems import BagOfWordsRater, VADERLexiconRater, NaiveBayesRater, MarkovChainRater, BERTRater, GPTRater, load_data

app = Flask(__name__)
CORS(app)

# Load the data for the rating systems
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()

raters = {
    "BoW": BagOfWordsRater(),
    "VADER": VADERLexiconRater(),
    "Naive Bayes": NaiveBayesRater(train_data, train_labels),
    "Markov Chains": MarkovChainRater(train_data, train_labels),
    "BERT": BERTRater(),
    "GPT": GPTRater(),
}

@app.route('/api/rate', methods=['POST'])
def rate_pickup_line():
    data = request.json
    system = data.get('system')
    line = data.get('line')

    if not system or not line:
        return jsonify({"error": "System and line are required"}), 400

    rater = raters.get(system)
    if not rater:
        return jsonify({"error": "Invalid rating system"}), 400

    try:
        rating = rater.rate_pickup_line(line)
        return jsonify({"rating": rating})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rate-all', methods=['POST'])
def rate_all_systems():
    data = request.json
    line = data.get('line')

    if not line:
        return jsonify({"error": "Pickup line is required"}), 400

    try:
        ratings = {system: rater.rate_pickup_line(line) for system, rater in raters.items()}
        return jsonify({"ratings": ratings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
