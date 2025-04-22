#!/bin/bash

echo "📦 Setting up virtual environment..."

# Check if venv exists
if [ ! -d "genre-env" ]; then
  python3 -m venv genre-env
  echo "✅ Virtual environment created: genre-env"
fi

# Activate the virtual environment
source genre-env/Scripts/activate
echo "✅ Virtual environment activated"

# Install required packages
echo "📚 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirments.txt

# Download NLTK data
echo "📥 Downloading necessary NLTK data..."
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Ask user if they want to regenerate the model
echo ""
read -p "🔁 Do you want to retrain the model using modelExp.ipynb? (y/N): " retrain

if [[ "$retrain" == "y" || "$retrain" == "Y" ]]; then
    echo "⚙️  Please open modelExp.ipynb and run all cells to retrain the model and regenerate .pkl files."
    echo "💡 After that, run this script again and skip retraining."
    exit 0
fi

# Run the Flask server
echo ""
echo "🚀 Launching Flask server..."
python3 app.py

echo ""
echo "🌐 Now open index.html in your browser to use the Genre Classifier!"
