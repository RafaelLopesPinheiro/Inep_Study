import os
import requests
import time
from zipfile import ZipFile
from tqdm import tqdm

def download_and_extract_inep_data(url: str, output_dir: str = "data/raw", zip_name: str = "inep_censo.zip"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, zip_name)

        print(f"📥 Starting download from:\n{url}")
        start_time = time.time()

        # Start request
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # Show progress
        with open(zip_path, "wb") as file, tqdm(
            desc="⬇️ Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=80
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        # Extraction
        print("📂 Extracting zip file...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        elapsed = time.time() - start_time
        print(f"✅ Download and extraction completed in {elapsed:.2f} seconds.")
        os.remove(zip_path)

    except requests.exceptions.RequestException as req_err:
        print(f"❌ Network error during download: {req_err}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    URL = "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip"
    download_and_extract_inep_data(URL)
