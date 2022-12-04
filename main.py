from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from typing import Optional


# initialize stopwords and spacy module
spacy.load('en_core_web_sm')
nlp = English()

# custom stopwords
STOP = set(STOP_WORDS)
STOP.update(set(punctuation))
STOP.update(['”', '“', '’', "'"])


nltk.download('stopwords')
nltk.download('punkt')

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class NltkSummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        """
        Initialize the text summarizer.
        Words that have a frequency term lower than min_cut
        or higher than max_cut will be ignored.
        """
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    def _compute_frequencies(self, word_sent):
        """
        Compute the frequency of each of word.
        Input:
        word_sent, a list of sentences already tokenized.
        Output:
        freq, a dictionary where freq[w] is the frequency of w.
        """
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        # frequencies normalization and fitering
        m = float(max(freq.values()))
        for w in list(freq.keys()):
            freq[w] = freq[w] / m
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, text, n):
        """
        Return a list of n sentences
        which represent the summary of text.
        """
        sents = sent_tokenize(text)
        assert n <= len(sents)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sent):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sents_idx = self._rank(ranking, n)
        return [sents[j] for j in sents_idx]

    def _rank(self, ranking, n):
        """ return the first n sentences with highest ranking """
        return nlargest(n, ranking, key=ranking.get)

def word_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

class SpacySummerizer:
    def __init__(self, min_freq=0.1):
        self._min_freq = min_freq
        self._word_count = None
        self._words = None

    def _get_word_count(self, text):
        words = word_tokenizer(text)
        word_count = {}
        for word in words:
            if word not in STOP:
                if word_count.get(word):
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        return word_count

    def _get_words(self, text):
        words = word_tokenizer(text)
        return [word for word in words if word not in STOP]

    def fit(self, text):
        self._word_count = self._get_word_count(text)
        self._words = self._get_words(text)
        self._sentences = text.split('.')
        return self

    def summarize(self, text, n):
        self.fit(text)

        word_freq = {}
        for word, count in self._word_count.items():
            word_freq[word] = count / float(len(self._words))
        word_sentence_map = {}
        for sentence in self._sentences:
            for word, freq in word_freq.items():
                if word in word_tokenizer(sentence.lower()):
                    if word not in word_sentence_map:
                        word_sentence_map[word] = []
                    word_sentence_map[word].append(sentence)

        sentence_score = {}
        for sentence in self._sentences:
            for word in word_tokenizer(sentence.lower()):
                if word in word_freq:
                    if sentence not in sentence_score:
                        sentence_score[sentence] = word_freq[word]
                    else:
                        sentence_score[sentence] += word_freq[word]
        top = nlargest(n, sentence_score, key=sentence_score.get)
        return [sentence for sentence in top]

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/nltk")
async def summarize(request:Request,text: str = Form(...), n: int = Form(...)):
    summarizer = NltkSummarizer()
    summary = summarizer.summarize(text, n)
    return summary

@app.post("/spacy")
async def summarize(request:Request, text: str = Form(...), n: int = Form(...)):
    summarizer = SpacySummerizer()
    summary = summarizer.summarize(text, n)
    return summary