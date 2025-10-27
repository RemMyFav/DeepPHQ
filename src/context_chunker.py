import csv
import re

### TODO: This framework makes a lot of ungrounded assumptions about the format of the CSVs. These WILL NEED to be reviewed once database access is obtained.


def clean_text(text: str) -> str:
    """Remove all text between angle brackets (<...>) and strip out all 'xxx'."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"xxx", "", text)
    return text.strip()

def read_csv_sentences(path: str):
    """Read a CSV containing sentences separated by commas."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        sentences = []
        for row in reader:
            for cell in row:
                clean_cell = clean_text(cell)
                if clean_cell:
                    sentences.append(clean_cell)
        return sentences


def strip_interview_questions(sentences):
    # TODO: Need to see the actual format of the csvs before this can be implemented
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
        words.append(s.split())

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
    input_file = "input.csv"
    clean_csv(input_file)
