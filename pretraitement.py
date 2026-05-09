import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
# Stopwords NLTK sans les mots de négation
negation_words = {"not", "no", "nor", "never", "neither", "nothing", "nowhere", "hardly", "barely", "scarcely"}
stop_words = set(stopwords.words('english')) - negation_words

def pretraitement(text):
    # retirer les html étiquettes par la librairie BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()

    # en miniscule
    text = text.lower()

    # supprimer les ponctuations et les symboles spéciales
    text = re.sub(r'[^a-z\s]', '', text)

    # enlever les stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return ' '.join(words)
