import pymupdf
import sys
import requests
import os


def get_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:  # iterate the document pages
        text += page.get_text()  # get plain text encoded as UTF-8
    return text


def fetch_and_extract(pdf_url, local_filename="tmp.pdf"):
    local_filename = pdf_url.split("/")[-1] + ".pdf"
    with requests.get(pdf_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    doc = pymupdf.open(local_filename)
    text = ""
    for page in doc:  # iterate the document pages
        text += page.get_text()  # get plain text encoded as UTF-8
    os.remove(local_filename)
    return text


if __name__ in "__main__":
    text = fetch_and_extract(sys.argv[1])
    word_count = len(text.split())
    print(f"word count: {word_count}")
    input_tokens = int((word_count / 750) * 1000)
    print(f"input tokens: {input_tokens}")
    if input_tokens < 200_000:
        cost = (1.25 / 1_000_000) * input_tokens
    else:
        cost = (2.5 / 1_000_000) * input_tokens
    print(f"input cost: {cost:.2f}")
