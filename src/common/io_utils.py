import json
from pathlib import Path


def read_json(path):
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def ensure_parent_dir(path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


def safe_slug(text):
    cleaned = str(text).strip().lower()

    replacements = {
        " ": "-",
        "/": "-",
        "\\": "-",
        ":": "-",
        ".": "",
        ",": "",
        "'": "",
        "\"": "",
        "(": "",
        ")": "",
    }

    for old_value, new_value in replacements.items():
        cleaned = cleaned.replace(old_value, new_value)

    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")

    return cleaned.strip("-")
