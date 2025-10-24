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

def download_daicwoz(
    output_dir: str = "data/raw/zips",
    base_url: str = "https://dcapswoz.ict.usc.edu/wwwdaicwoz",
    start_id: int = 300,
    end_id: int = 492,
    skip_existing: bool = True
):
    """
    Download all DAIC-WOZ participant ZIP files (no authentication).

    Parameters
    ----------
    output_dir : str
        Directory to save ZIP files (default: data/raw/zips)
    base_url : str
        Base URL of DAIC-WOZ dataset
    start_id : int
        Starting participant ID (default: 300)
    end_id : int
        Ending participant ID (default: 492)
    skip_existing : bool
        If True, skip downloading files that already exist
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    for pid in range(start_id, end_id + 1):
        filename = f"{pid}_P.zip"
        url = f"{base_url}/{filename}"
        output_path = os.path.join(output_dir, filename)

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping existing file: {filename}")
            continue

        print(f"Downloading {filename} ...")

        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"File not found: {filename}")
                    continue
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

        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    print("All downloads attempted. Check data/raw/zips/")

def extract_daicwoz_transcripts(
    zip_dir: str = "data/raw/zips",
    out_dir: str = "data/raw/transcripts",
    remove_zip: bool = False
):
    """
    Extracts the *_TRANSCRIPT.csv file from each participant ZIP archive
    and saves it to a target directory.

    Parameters
    ----------
    zip_dir : str
        Path to the directory containing all participant ZIP archives
        (e.g., `data/raw/zips`).
    out_dir : str
        Path to the output directory where transcript CSVs will be saved.
    remove_zip : bool
        If True, deletes each ZIP file after successful extraction
        to save disk space.
    """
    os.makedirs(out_dir, exist_ok=True)
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
    print(f"Found {len(zip_files)} zip files in {zip_dir}")

    for fname in zip_files:
        zip_path = os.path.join(zip_dir, fname)
        pid = fname.split("_")[0]  # Extract participant ID (e.g., 300 from 300_P.zip)

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                # Identify transcript files within the archive
                transcript_files = [
                    n for n in z.namelist() if "TRANSCRIPT" in n and n.endswith(".csv")
                ]
                if not transcript_files:
                    print(f"No transcript found in {fname}")
                    continue

                # Expect exactly one transcript per archive
                transcript_name = transcript_files[0]
                output_path = os.path.join(out_dir, f"{pid}_TRANSCRIPT.csv")

                # Extract file temporarily
                z.extract(transcript_name, out_dir)
                extracted_path = os.path.join(out_dir, transcript_name)

                # Handle nested folders (move transcript to output root)
                if extracted_path != output_path:
                    shutil.move(extracted_path, output_path)
                    parent = os.path.dirname(extracted_path)
                    if parent != out_dir and os.path.exists(parent):
                        shutil.rmtree(parent, ignore_errors=True)

                print(f"Extracted: {pid}_TRANSCRIPT.csv")

            # Optionally remove the zip after successful extraction
            if remove_zip:
                os.remove(zip_path)
                print(f"Removed: {fname}")

        except zipfile.BadZipFile:
            print(f"Corrupted zip file: {fname}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    print(f"All transcripts extracted to {out_dir}")

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