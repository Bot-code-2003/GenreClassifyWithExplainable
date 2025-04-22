import joblib, re, nltk, io, base64
import matplotlib
matplotlib.use('Agg')  # ✅ Fixes GUI issue for Flask
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)
CORS(app)

# Load model/vectorizer
model = joblib.load("saved_model.pkl")
vectorizer = joblib.load("saved_vectorizer.pkl")

genre_mapping = {
    0: "crime", 1: "fantasy", 2: "history", 3: "horror",
    4: "psychology", 5: "romance", 6: "science",
    7: "sports", 8: "thriller", 9: "travel"
}

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w not in stopwords])

class GenrePredictor:
    def __init__(self, model, vectorizer, mapping):
        self.model = model
        self.vectorizer = vectorizer
        self.mapping = mapping
        self.class_names = [mapping[i] for i in range(len(mapping))]

    def predict(self, text):
        clean = preprocess(text)
        vec = self.vectorizer.transform([clean])
        probs = self.model.predict_proba(vec)[0]
        label = self.model.predict(vec)[0]
        return {
            "processed_text": clean,
            "predicted_label": label,
            "predicted_genre": self.mapping[label],
            "prediction_probabilities": probs
        }

    def predict_proba(self, texts):
        return self.model.predict_proba(self.vectorizer.transform(texts))

predictor = GenrePredictor(model, vectorizer, genre_mapping)
explainer = LimeTextExplainer(class_names=predictor.class_names)

@app.route("/")
def home():
    return jsonify({
        "message": "🎉 Genre Classifier API is running!",
        "endpoints": {
            "predict": "POST /predict",
            "explain": "POST /explain (shows chart)",
            "genres": "GET /genres",
            "health": "GET /health"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/genres")
def genres():
    return jsonify(genre_mapping)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    result = predictor.predict(data["text"])
    probs = {genre_mapping[i]: float(p) for i, p in enumerate(result["prediction_probabilities"])}
    return jsonify({
        "predicted_genre": result["predicted_genre"],
        "confidence": float(result["prediction_probabilities"][result["predicted_label"]]),
        "prediction_probabilities": probs
    })

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = predictor.predict(data["text"])
    predicted_label = result["predicted_label"]
    predicted_genre = result["predicted_genre"]

    # Generate LIME explanations for top 4 genres
    explanation = explainer.explain_instance(
        result["processed_text"],
        predictor.predict_proba,
        top_labels=9,
        num_features=10  # Reduced for compact display
    )

    # Get top 4 genres by probability
    genre_scores = sorted(
        enumerate(result["prediction_probabilities"]),
        key=lambda x: x[1],
        reverse=True
    )[:9]

    # Generate plots and collect features for each genre
    plot_data = []
    for label, score in genre_scores:
        # Generate plot
        fig = explanation.as_pyplot_figure(label=label)
        fig.set_size_inches(6, 3)  # Compact size for grid
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)

        # Get word contributions
        features = explanation.as_list(label=label)

        plot_data.append({
            "genre": genre_mapping[label],
            "score": score * 100,
            "plot": img_base64,
            "features": features
        })

    # Return enhanced HTML
    return render_template_string('''
    <!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genre Explanation</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --primary: #8b5cf6;
            --success: #10b981;
            --danger: #ef4444;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            margin: 0 auto;
            padding: 2rem;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .card-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 1.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .genre-badge {
            background: linear-gradient(135deg, var(--primary), #3b82f6);
            color: white;
            padding: 0.3rem 1rem;
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: capitalize;
        }

        .card-body {
            padding: 1.5rem;
        }

        .confidence {
            font-size: 0.875rem;
            background: rgba(16, 185, 129, 0.1);
            padding: 0.3rem 0.75rem;
            color: var(--success);
            font-weight: 500;
            margin-bottom: 1.5rem;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .comparison-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border: 1px solid var(--border-color);
        }

        .comparison-title {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.75rem;
            text-align: center;
            color: var(--text-primary);
        }

        .comparison-title span {
            color: var(--success);
        }

        .plot-container {
            text-align: center;
            margin-bottom: 1rem;
        }

        .plot-container img {
            width: 400px;
            max-width: 100%;
            min-height: 300px;
            border: 1px solid var(--border-color);
        }

        .features-section {
            margin-top: 0.5rem;
        }

        .features-title {
            font-weight: 500;
            font-size: 0.9rem;
            padding: 0.5rem 0;
            color: var(--text-primary);
        }

        .features-list {
            list-style: none;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.03);
        }

        .feature-item {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            font-size: 0.85rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .feature-word {
            font-weight: 500;
        }

        .feature-score {
            color: var(--text-secondary);
        }

        .feature-score.positive {
            color: var(--success);
        }

        .feature-score.negative {
            color: var(--danger);
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        footer a {
            color: var(--primary);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .card-title {
                font-size: 1.5rem;
            }

            .comparison-grid {
                grid-template-columns: 1fr;
            }

            .plot-container img {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-header">
                <h1 class="card-title">
                    <span>🧠</span> Genre Explanation
                </h1>
                <div class="genre-badge">{{ genre }}</div>
            </div>
            <div class="card-body">
                <div class="confidence">Confidence: {{ confidence }}%</div>
                <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">
                    📊 Genre Comparison
                </h3>
                <div class="comparison-grid">
                    {% for data in plot_data %}
                    <div class="comparison-card">
                        <div class="comparison-title">
                            {{ "Why" if data.genre == genre else "Why not" }}
                            <span>{{ data.genre }}</span>
                            ({{ "%.2f"|format(data.score) }}%)
                        </div>
                        <div class="plot-container">
                            <img src="data:image/png;base64,{{ data.plot }}" alt="{{ data.genre }} Explanation">
                        </div>
                        <div class="features-section">
                            <div class="features-title">Word Contributions for {{ data.genre }}</div>
                            <ul class="features-list">
                                {% for word, score in data.features %}
                                <li class="feature-item">
                                    <span class="feature-word">{{ word }}</span>
                                    <span class="feature-score {% if score > 0 %}positive{% else %}negative{% endif %}">
                                        {{ "%.4f"|format(score) }}
                                    </span>
                                </li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    ''', genre=predicted_genre,
         confidence="%.2f" % (result["prediction_probabilities"][predicted_label] * 100),
         plot_data=plot_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
