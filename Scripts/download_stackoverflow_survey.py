# download_stackoverflow_survey.py

import os
import re
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import zipfile
from urllib.parse import urljoin

# Step 1: User input
year = input("Enter the year of the Stack Overflow survey you want to download (e.g., 2024): ").strip()
base_url = "https://survey.stackoverflow.co/"
headers = {"User-Agent": "Mozilla/5.0"}

# Step 2: Scrape main page for ZIP file
try:
    res = requests.get(base_url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
except requests.exceptions.RequestException as e:
    print(f"Could not load the base page: {e}")
    exit()

# Step 3: Find the ZIP file link
zip_url = None
for link in soup.find_all("a", href=True):
    href = link["href"]
    if href.endswith(".zip") and year in href:
        zip_url = href if href.startswith("http") else urljoin(base_url, href)
        print(f"Found ZIP file: {zip_url}")
        break

if not zip_url:
    print(f"No ZIP file found for year {year}.")
    exit()

# Step 4: Download ZIP
try:
    zip_response = requests.get(zip_url)
    zip_response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error downloading ZIP: {e}")
    exit()

# Step 5: Prepare folders
raw_folder = r"C:\Users\kanmani\Desktop\AutoStack360\Data\Raw"
meta_folder = r"C:\Users\kanmani\Desktop\AutoStack360\Data\Metadata"
os.makedirs(raw_folder, exist_ok=True)
os.makedirs(meta_folder, exist_ok=True)

# Step 6: Extract and organize
try:
    with zipfile.ZipFile(BytesIO(zip_response.content)) as z:
        print("Files in ZIP:", z.namelist())
        for file_name in z.namelist():
            if "survey_results_public.csv" in file_name:
                save_path = os.path.join(raw_folder, f"{year}_survey_results_public.csv")
            else:
                save_path = os.path.join(meta_folder, f"{year}_" + os.path.basename(file_name))

            with z.open(file_name) as src, open(save_path, "wb") as dest:
                dest.write(src.read())
            print(f"Extracted: {save_path}")

except Exception as e:
    print(f"Error extracting ZIP: {e}")
