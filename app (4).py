import streamlit as st
import subprocess
import sys

# Install dependencies FIRST (before any imports)
def install_dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "scikit-learn", "pickle5"])
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        st.error(f"Dependency installation failed: {e}")

install_dependencies()

# Now import everything else
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import random

# Rest of your original code continues here...
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
# Function to download NLTK data with error handling
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

# Download required NLTK data at startup
download_nltk_data()

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        with open('fitbot_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models['vectorizer'], models['nb_model']
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text):
        try:
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            processed = []
            for word, tag in tagged:
                if word.isalpha() and word not in self.stop_words:
                    pos = tag[0].lower()
                    pos = pos if pos in ['a', 'r', 'n', 'v'] else 'n'
                    processed.append(self.lemmatizer.lemmatize(word, pos))
            return ' '.join(processed)
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
            return text.lower()  # fallback to simple lowercase

def generate_response(query, label):
    workout_db = {
        'abs': ["Plank (3 sets of 30-60 seconds)", "Crunches (3 sets of 15-20 reps)", 
                "Leg raises (3 sets of 12-15 reps)", "Russian twists (3 sets of 20 reps)"],
        'chest': ["Push-ups (4 sets to failure)", "Bench press (4 sets of 8-12 reps)",
                 "Dumbbell flyes (3 sets of 12 reps)", "Chest dips (3 sets to failure)"],
        'general': ["Full body workout 3x/week with compound movements", 
                   "Try alternating cardio and strength days",
                   "Start with bodyweight exercises and progress to weights"]
    }
    
    diet_db = {
        'weight loss': ["Calorie deficit of 300-500 calories/day", 
                        "High protein, moderate fat, low carb",
                        "Focus on whole foods and vegetables"],
        'muscle gain': ["Calorie surplus of 200-500 calories/day",
                       "1g protein per pound of body weight",
                       "Eat every 2-3 hours"],
        'general': ["Stay hydrated with 2-3L water daily",
                   "Balance macros (40% carbs, 30% protein, 30% fat)",
                   "Include variety of colorful vegetables"]
    }
    
    gym_db = {
        'equipment': ["Always wipe down equipment after use", 
                     "Ask staff for orientation on machines",
                     "Start with light weights to learn form"],
        'etiquette': ["Don't hog multiple machines at once",
                     "Re-rack your weights after use",
                     "Keep phone usage minimal"],
        'general': ["Peak hours are usually 5-8pm, try off-peak times",
                   "Most gyms offer free orientation sessions",
                   "Bring a lock for the locker room"]
    }
    
    try:
        tokens = word_tokenize(query.lower())
        keywords = []
        for word in tokens:
            if word in ['abs', 'chest', 'back', 'legs', 'arms']:
                keywords.append(word)
            elif word in ['lose', 'loss', 'weight']:
                keywords.append('weight loss')
            elif word in ['gain', 'muscle', 'mass']:
                keywords.append('muscle gain')
            elif word in ['equipment', 'machine', 'rack']:
                keywords.append('equipment')
            elif word in ['etiquette', 'manners', 'rules']:
                keywords.append('etiquette')
        
        if label == 'workout':
            responses = workout_db.get('general', [])
            for kw in keywords:
                responses.extend(workout_db.get(kw, []))
            return random.choice(responses) if responses else "I recommend focusing on compound movements like squats and deadlifts."
        
        elif label == 'diet':
            responses = diet_db.get('general', [])
            for kw in keywords:
                responses.extend(diet_db.get(kw, []))
            return random.choice(responses) if responses else "A balanced diet with protein, carbs, and healthy fats is essential."
        
        elif label == 'gym':
            responses = gym_db.get('general', [])
            for kw in keywords:
                responses.extend(gym_db.get(kw, []))
            return random.choice(responses) if responses else "Remember to wipe down equipment after use and respect others' space."
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I encountered an error processing your request. Please try again."

def main():
    st.title("FitBot - Your AI Fitness Assistant")
    st.write("Ask me about workouts, nutrition, or gym advice!")
    
    vectorizer, nb_model = load_models()
    
    if vectorizer is None or nb_model is None:
        st.error("Failed to load the required models. Please check if 'fitbot_models.pkl' exists.")
        return
    
    preprocessor = TextPreprocessor()
    
    user_input = st.text_input("What would you like to know about fitness, nutrition, or gyms?")
    
    if user_input:
        with st.spinner('Processing your question...'):
            try:
                processed = preprocessor.preprocess(user_input)
                vec = vectorizer.transform([processed])
                pred = nb_model.predict(vec)[0]
                
                response = generate_response(user_input, pred)
                
                st.subheader("FitBot says:")
                st.write(response)
                
                st.write(f"(This was classified as a {pred} question)")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
