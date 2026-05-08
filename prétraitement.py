import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

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
