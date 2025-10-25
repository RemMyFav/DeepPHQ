"""
download_daicwoz.py
-------------------
Batch-download all DAIC-WOZ participant ZIP files (no authentication required).

Usage (from notebook):
    from src.download_daicwoz import download_daicwoz
    download_daicwoz(start_id=300, end_id=384)
"""

import os
import requests
from tqdm import tqdm
import zipfile
import shutil


def extract_daicwoz_transcripts(
    zip_dir: str = "../data/raw/zips",
    out_dir: str = "../data/raw/transcripts",
    base_url: str = "https://dcapswoz.ict.usc.edu/wwwdaicwoz",
    start_id: int = 300,
    end_id: int = 301,
    remove_zip: bool = True
):
    """
    Ensures all DAIC-WOZ participant transcripts are extracted and available.

    This function checks for existing *_TRANSCRIPT.csv files and skips them.
    If a participant's transcript is missing but their ZIP exists, it extracts
    from the ZIP. If both are missing, it downloads the ZIP from the DAIC-WOZ
    server and extracts the transcript. Designed for idempotent use in pipelines.

    Parameters
    ----------
    zip_dir : str
        Directory containing downloaded participant ZIP files.
    out_dir : str
        Output directory where transcript CSVs will be saved.
    base_url : str
        Base URL of the DAIC-WOZ dataset.
    start_id : int
        Starting participant ID (inclusive).
    end_id : int
        Ending participant ID (inclusive).
    remove_zip : bool
        If True, deletes ZIP files after successful extraction.
    """

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)

    zip_files = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
    existing_transcripts = {f.split("_")[0] for f in os.listdir(out_dir) if f.endswith("_TRANSCRIPT.csv")}
    print(f"Found {len(zip_files)} zip files and {len(existing_transcripts)} transcripts.\n")

    for pid in range(start_id, end_id + 1):
        pid_str = str(pid)
        zip_name = f"{pid_str}_P.zip"
        zip_path = os.path.join(zip_dir, zip_name)
        transcript_name = f"{pid_str}_TRANSCRIPT.csv"
        transcript_path = os.path.join(out_dir, transcript_name)
        url = f"{base_url}/{zip_name}"

        # --- 1. Skip if transcript already exists ---
        if os.path.exists(transcript_path):
            print(f"Transcript already exists for {pid_str}, skipping.")
            continue

        # --- 2. Extract if ZIP already exists ---
        if os.path.exists(zip_path):
            print(f"Extracting from existing ZIP for {pid_str}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    transcript_files = [n for n in z.namelist() if "TRANSCRIPT" in n and n.endswith(".csv")]
                    if not transcript_files:
                        print(f"No transcript found in {zip_name}")
                        continue
                    z.extract(transcript_files[0], out_dir)
                    extracted_path = os.path.join(out_dir, transcript_files[0])
                    if extracted_path != transcript_path:
                        shutil.move(extracted_path, transcript_path)
                    print(f"Extracted: {transcript_name}")
                if remove_zip:
                    os.remove(zip_path)
            except Exception as e:
                print(f"Failed to extract {zip_name}: {e}")
            continue

        # --- 3. Download ZIP then extract ---
        print(f"Downloading {zip_name} ...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"File not found on server: {zip_name}")
                    continue
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zip_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=zip_name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            print(f"Downloaded: {zip_name}")

            with zipfile.ZipFile(zip_path, "r") as z:
                transcript_files = [n for n in z.namelist() if "TRANSCRIPT" in n and n.endswith(".csv")]
                if not transcript_files:
                    print(f"No transcript found in {zip_name}")
                    continue
                z.extract(transcript_files[0], out_dir)
                extracted_path = os.path.join(out_dir, transcript_files[0])
                if extracted_path != transcript_path:
                    shutil.move(extracted_path, transcript_path)
                print(f"Extracted: {transcript_name}")
            if remove_zip:
                os.remove(zip_path)

        except Exception as e:
            print(f"Failed to download or extract {zip_name}: {e}")

    print(f"\nAll transcripts ensured in {out_dir}.")

def download_phq_file(
    filename: str = "full_test_split.csv",
    output_dir: str = "data/raw",
    base_url: str = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"
):
    """
    Downloads the metadata CSV file (e.g., full_test_split.csv) from the DAIC-WOZ dataset website.

    Parameters
    ----------
    filename : str
        The file name to download (default: "full_test_split.csv").
    output_dir : str
        Directory where the file will be saved (default: "data/raw").
    base_url : str
        Base URL of the DAIC-WOZ dataset server.
    """

    os.makedirs(output_dir, exist_ok=True)
    url = f"{base_url}/{filename}"
    output_path = os.path.join(output_dir, filename)

    # Skip existing file
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading {filename} from {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                print("File not found on server.")
                return None
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            with open(output_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        print(f"Downloaded: {filename}")
        return output_path

    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None