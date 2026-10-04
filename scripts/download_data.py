from pathlib import Path

import requests

# Zenodo-hosted dataset. If Zenodo changes the direct file URL, use the record page
# documented in README and update this constant.
URL = "https://zenodo.org/records/19161757/files/dreamviews-posts.tsv?download=1"
OUT = Path("data/raw/dreamviews-posts.tsv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 1_000_000:
        print(f"Already present: {OUT}")
        return
    print(f"Downloading to {OUT} ...")
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with OUT.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Done: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
