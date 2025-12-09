import csv
import re
import random
import pandas as pd
import os
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove all text between angle brackets (<...>) and strip out all 'xxx'."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"xxx", "", text)
    return text.strip()

def read_csv_sentences(path: str):
    """
    Robust reader for DAIC-WOZ transcripts.
    Automatically handles:
        - encoding issues
        - unknown delimiters
        - missing or weird column names
    Returns participant sentences.
    """

    # Try encodings
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                sample = f.read(2000)
                f.seek(0)

                # Detect delimiter
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
                except:
                    dialect = csv.excel_tab  # fallback = tab

                reader = csv.DictReader(f, delimiter=dialect.delimiter)

                # handle cases where column names have BOM or spaces
                fixed_rows = []
                for row in reader:
                    clean_row = {k.strip().lower(): v for k, v in row.items()}
                    fixed_rows.append(clean_row)

                sentences = []
                for row in fixed_rows:
                    if row.get("speaker", "").lower() == "participant":
                        text = row.get("value", "").strip()
                        if text:
                            sentences.append(clean_text(text))
                return sentences

        except:
            continue

    print(f"[!] Failed to parse transcript: {path}")
    return []

def word_level_aug(sentences, N):
    """
    Pure random bag-of-words sampling.
    Randomly sample N words from *all words* in the transcript.
    """
    words = " ".join(sentences).split()
    if not words:
        return ""

    # If there are fewer words than N → repeat then trim
    if len(words) < N:
        repeat = (N // len(words)) + 1
        words = (words * repeat)

    sampled = random.sample(words, N)
    return " ".join(sampled)


def sentence_level_aug(sentences, N):
    """
    Sentence-level augmentation:
    - Randomly sample full sentences (with replacement)
    - Only whole sentences are added (never cut any sentence)
    - If adding a sentence would exceed N words → stop and pad the rest
    - Always returns EXACTLY N words
    """

    if not sentences:
        return " ".join(["<PAD>"] * N)

    collected = []
    total_words = 0

    while True:
        s = random.choice(sentences)
        w = s.split()
        L = len(w)

        # If adding this full sentence would exceed N → stop and pad
        if total_words + L > N:
            break

        # Otherwise add this whole sentence
        collected.extend(w)
        total_words += L

        # If we reached exactly N, stop
        if total_words == N:
            break

    # Pad to exactly N words
    if total_words < N:
        collected.extend(["<PAD>"] * (N - total_words))

    return " ".join(collected[:N])

def dialogue_level_aug(sentences, N, min_future_words=128):
    """
    Dialogue-level augmentation:
    - choose random sentence index as start
    - ensure that from this start, we have >= min_future_words real words
    - accumulate forward sequentially
    - pad/truncate to EXACT N
    """

    if not sentences:
        return " ".join(["<PAD>"] * N)

    # Convert to words for future-length check
    all_words = " ".join(sentences).split()
    L = len(all_words)

    # Build cumulative word counts so we can know
    # how many words remain after each sentence index
    sentence_word_lens = [len(s.split()) for s in sentences]
    cum_words = []
    total = 0
    for ln in sentence_word_lens:
        total += ln
        cum_words.append(total)

    # Try a few times to find a valid starting point
    for _ in range(20):
        start_idx = random.randint(0, len(sentences) - 1)

        # remaining words after this sentence
        words_before = cum_words[start_idx] - sentence_word_lens[start_idx]
        words_after = L - words_before

        if words_after >= min_future_words:
            break
    else:
        # fallback: no valid start → just use start 0
        start_idx = 0

    # accumulate forward
    collected = []
    for s in sentences[start_idx:]:
        collected.extend(s.split())
        if len(collected) >= N:
            break

    # pad
    if len(collected) < N:
        collected.extend(["<PAD>"] * (N - len(collected)))

    return " ".join(collected[:N])

def match_phq_transcripts(
    transcript_dir="data/raw/transcripts",
    meta_csv="data/raw/full_test_split.csv"
):
    """
    Build a mapping: participant_id -> PHQ score.
    Only returns IDs for which transcripts exist.
    """

    # --- Load metadata ---
    meta_df = pd.read_csv(meta_csv)
    meta_df["Participant_ID"] = meta_df["Participant_ID"].astype(int)
    phq_lookup = dict(zip(meta_df["Participant_ID"], meta_df["PHQ_8Total"]))

    # --- Find all transcript IDs available ---
    transcript_ids = []
    for file in Path(transcript_dir).glob("*.csv"):
        pid = int(re.search(r"\d+", file.stem).group())
        transcript_ids.append(pid)

    transcript_ids = sorted(set(transcript_ids))

    # --- Build aligned mapping ---
    aligned = {}
    missing = []

    for pid in transcript_ids:
        if pid in phq_lookup:
            aligned[pid] = phq_lookup[pid]
        else:
            aligned[pid] = -1       # mark missing label
            missing.append(pid)

    print(f"Loaded PHQ mapping for {len(aligned)} participants.")
    if missing:
        print(f"⚠ Missing PHQ score for: {missing}")

    return aligned
def match_all_phq_transcripts(
    transcript_dir="data/raw/transcripts",
    meta_csv="data/raw/full_test_split.csv"
):
    """
    Build a mapping: participant_id -> 8 PHQ item scores (array of length 8).
    Only returns IDs for which transcripts exist.
    """

    # 1. Load metadata
    meta_df = pd.read_csv(meta_csv)
    meta_df["Participant_ID"] = meta_df["Participant_ID"].astype(int)

    # Define 8 item columns
    item_cols = [
        "PHQ_8NoInterest",
        "PHQ_8Depressed",
        "PHQ_8Sleep",
        "PHQ_8Tired",
        "PHQ_8Appetite",
        "PHQ_8Failure",
        "PHQ_8Concentrating",
        "PHQ_8Moving",
    ]

    # Build lookup: pid → array(8,)
    phq_lookup = {}
    for _, row in meta_df.iterrows():
        pid = int(row["Participant_ID"])
        item_vec = row[item_cols].values.astype(float)  # float array length 8
        phq_lookup[pid] = item_vec

    # 2. Find all transcript IDs
    transcript_ids = []
    for file in Path(transcript_dir).glob("*.csv"):
        pid = int(re.search(r"\d+", file.stem).group())
        transcript_ids.append(pid)

    transcript_ids = sorted(set(transcript_ids))

    # 3. Build aligned mapping
    aligned = {}
    missing = []

    for pid in transcript_ids:
        if pid in phq_lookup:
            aligned[pid] = phq_lookup[pid]  # (8,) vector
        else:
            aligned[pid] = np.array([-1]*8, dtype=float)
            missing.append(pid)

    print(f"Loaded PHQ mapping for {len(aligned)} participants.")
    if missing:
        print(f"⚠ Missing PHQ labels for: {missing}")

    return aligned

def generate_dataset(
    transcript_dir: str,
    phq_dict: dict
):
    """
    Load all transcripts under a folder and match them with PHQ scores 
    from phq_dict {pid: score}.

    Returns:
        all_transcripts = [
            (pid, [sentence1, sentence2, ...], phq_score),
            ...
        ]
    """

    all_transcripts = []

    for file in Path(transcript_dir).glob("*.csv"):
        # --- Extract participant ID from file name ---
        pid = int(re.search(r"\d+", file.stem).group())

        # --- Skip if this participant not in PHQ dict ---
        if pid not in phq_dict:
            print(f"[!] PID {pid} not found in PHQ dict, skipping.")
            continue

        # --- Read participant-only sentences ---
        sentences = read_csv_sentences(file)

        if not sentences:
            print(f"[!] Empty or malformed transcript for {pid}, skipping.")
            continue

        score = phq_dict[pid]

        all_transcripts.append((pid, sentences, score))

    # Sort for cleanliness
    all_transcripts.sort(key=lambda x: x[0])

    print(f"Loaded {len(all_transcripts)} transcripts with PHQ scores.")
    return all_transcripts

def generate_all_phq_dataset(
    transcript_dir: str,
    phq_dict: dict
):
    """
    Load all transcripts under a folder and match them with PHQ scores 
    from phq_dict {pid: score}.

    Returns:
        all_transcripts = [
            (pid, [sentence1, sentence2, ...], phq_score),
            ...
        ]
    """

    all_transcripts = []

    for file in Path(transcript_dir).glob("*.csv"):
        # --- Extract participant ID from file name ---
        pid = int(re.search(r"\d+", file.stem).group())

        # --- Skip if this participant not in PHQ dict ---
        if pid not in phq_dict:
            print(f"[!] PID {pid} not found in PHQ dict, skipping.")
            continue

        # --- Read participant-only sentences ---
        sentences = read_csv_sentences(file)

        if not sentences:
            print(f"[!] Empty or malformed transcript for {pid}, skipping.")
            continue

        score = phq_dict[pid]

        all_transcripts.append((pid, sentences, score))

    # Sort for cleanliness
    all_transcripts.sort(key=lambda x: x[0])

    print(f"Loaded {len(all_transcripts)} transcripts with PHQ scores.")
    return all_transcripts

def build_text_representations(all_transcripts, sequence_len=512, num_samples_per_pid=20):
    dataset_word = []
    dataset_sentence = []
    dataset_dialogue = []

    for pid, sentences, score in all_transcripts:

        for _ in range(num_samples_per_pid):

            # --- 1. word-level ---
            w = word_level_aug(sentences, sequence_len)
            dataset_word.append((pid, w, score))

            # --- 2. sentence-level ---
            s = sentence_level_aug(sentences, sequence_len)
            dataset_sentence.append((pid, s, score))

            # --- 3. dialogue-level ---
            d = dialogue_level_aug(sentences, sequence_len)
            dataset_dialogue.append((pid, d, score))

    return dataset_word, dataset_sentence, dataset_dialogue


def save_text_representations(
    dataset_word,
    dataset_sentence,
    dataset_dialogue,
    output_dir="data/processed"
):
    """
    Save the three levels (word / sentence / dialogue) into separate CSV files.
    
    Each dataset is a list of tuples: (pid, text, phq_score).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Word-level
    df_word = pd.DataFrame(dataset_word, columns=["PID", "Text", "PHQ_Score"])
    df_word.to_csv(os.path.join(output_dir, "word_level.csv"), index=False)
    print(f"Saved word-level dataset → {os.path.join(output_dir, 'word_level.csv')}")

    # Sentence-level
    df_sentence = pd.DataFrame(dataset_sentence, columns=["PID", "Text", "PHQ_Score"])
    df_sentence.to_csv(os.path.join(output_dir, "sentence_level.csv"), index=False)
    print(f"Saved sentence-level dataset → {os.path.join(output_dir, 'sentence_level.csv')}")

    # Dialogue-level
    df_dialogue = pd.DataFrame(dataset_dialogue, columns=["PID", "Text", "PHQ_Score"])
    df_dialogue.to_csv(os.path.join(output_dir, "dialogue_level.csv"), index=False)
    print(f"Saved dialogue-level dataset → {os.path.join(output_dir, 'dialogue_level.csv')}")
