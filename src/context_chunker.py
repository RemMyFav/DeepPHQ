import csv
import re
import random


def clean_text(text: str) -> str:
    """Remove all text between angle brackets (<...>) and strip out all 'xxx'."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"xxx", "", text)
    return text.strip()

def read_csv_sentences(path: str):
    """Read a TSV, returning only Participant dialogue."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("speaker", "").strip().lower() == "participant":
                value = row.get("value", "").strip()
                if value:
                    sentences.append(clean_text(value))
    return sentences


def word_level(sentences, N):
    """Sample N words randomly from all sentences."""
    '''
    # This version of the code will ensure unique words, not sure if that will be necessary for this 
    words = set()
    for s in sentences:
        words.update(s.split())
    words = list(words)
    '''
    words = list()
    for s in sentences:
        for w in s.split():
            words.append(w)

    if N > len(words):
        N = len(words)
    sampled = random.sample(words, N)

    return " ".join(sampled)


def sentence_level(sentences, N):
    """Take sentences in order until total words ≈ N (truncate last sentence)."""
    selected = []
    word_count = 0
    for s in sentences:
        words = s.split()
        if word_count + len(words) > N:
            remaining = N - word_count
            selected.append(" ".join(words[:remaining]))
            break
        selected.append(s)
        word_count += len(words)
        if word_count >= N:
            break
    return " ".join(selected)


def dialogue_level(sentences, N):
    """Take sentences in order until total words ≈ N (truncate last sentence)."""
    selected = []
    word_count = 0
    for s in sentences:
        words = s.split()
        if word_count + len(words) > N:
            remaining = N - word_count
            selected.append(" ".join(words[:remaining]))
            break
        selected.append(s)
        word_count += len(words)
        if word_count >= N:
            break
    return " ".join(selected)

if __name__ == "__main__":
    input_file = "../data/raw/transcripts/300_TRANSCRIPT.csv"
    sentences = read_csv_sentences(input_file)
    print(word_level(sentences, 512))
    print(sentence_level(sentences, 512))
    print(dialogue_level(sentences, 512))
