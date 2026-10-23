"""Prepare the two small, deterministic local data caches used by the notebook.

The source datasets remain Hugging Face datasets.  Local copies make the exact
Multi-News example and Tatoeba subset stable across notebook runs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "data" / "summarization"
LANGUAGE_DIR = ROOT / "data" / "language_id"

MULTI_NEWS_REPOSITORY = "mehdielg/multi_news"
MULTI_NEWS_SPLIT = "train"
MULTI_NEWS_INDEX = 32766

TATOEBA_REPOSITORY = "ymoslem/Tatoeba-Translations"
TATOEBA_REVISION = "c313a0934b5e2f50f78230fdedfe63ea36e1a05f"
TATOEBA_FILES = [
    f"https://huggingface.co/datasets/{TATOEBA_REPOSITORY}/resolve/"
    f"{TATOEBA_REVISION}/data/train-{part:05d}-of-00003.parquet"
    for part in range(3)
]
ISO3_TO_ISO2 = {
    "kaz": "kk",
    "rus": "ru",
    "eng": "en",
    "spa": "es",
    "por": "pt",
    "fra": "fr",
    "deu": "de",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def prepare_multi_news() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {MULTI_NEWS_REPOSITORY}:{MULTI_NEWS_SPLIT}[{MULTI_NEWS_INDEX}]...")
    stream = load_dataset(MULTI_NEWS_REPOSITORY, split=MULTI_NEWS_SPLIT, streaming=True)
    example = next(row for index, row in enumerate(stream) if index == MULTI_NEWS_INDEX)
    articles = [clean_text(part) for part in example["document"].split("|||||") if part.strip()]
    summary = re.sub(r"^[–-]\s*", "", clean_text(example["summary"]))

    article_counts = [len(article.split()) for article in articles]
    assert len(articles) == 5
    assert min(article_counts) >= 250
    assert 100 <= len(summary.split()) <= 150

    for index, article in enumerate(articles, start=1):
        (SUMMARY_DIR / f"article_{index}.txt").write_text(article + "\n", encoding="utf-8")
    (SUMMARY_DIR / "reference_summary.txt").write_text(summary + "\n", encoding="utf-8")
    metadata = {
        "requested_repository": "tau/multi_news",
        "actual_repository": MULTI_NEWS_REPOSITORY,
        "reason": "tau/multi_news returned a non-UTF-8 Google Drive response on 2026-10-23",
        "split": MULTI_NEWS_SPLIT,
        "stream_position_zero_based": MULTI_NEWS_INDEX,
        "article_word_counts": article_counts,
        "summary_word_count": len(summary.split()),
    }
    (SUMMARY_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metadata)


def prepare_tatoeba(per_language: int = 700) -> None:
    """Read remote parquet row groups until 700 unique examples/language are found."""

    LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LANGUAGE_DIR / "tatoeba_7_languages.csv"
    samples: dict[str, list[str]] = defaultdict(list)
    seen_global: set[str] = set()
    columns = ["lang_src", "sentence_src", "lang_tgt", "sentence_tgt"]

    for url in TATOEBA_FILES:
        print(f"Scanning {url.rsplit('/', 1)[-1]}...")
        with fsspec.open(url, "rb", block_size=8 * 1024 * 1024, cache_type="readahead") as handle:
            parquet = pq.ParquetFile(handle)
            for group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(group_index, columns=columns)
                frame = table.to_pandas()
                for side in ("src", "tgt"):
                    for lang3, sentence in zip(frame[f"lang_{side}"], frame[f"sentence_{side}"]):
                        if lang3 not in ISO3_TO_ISO2:
                            continue
                        language = ISO3_TO_ISO2[lang3]
                        sentence = clean_text(sentence)
                        key = sentence.casefold()
                        if sentence and key not in seen_global and len(samples[language]) < per_language:
                            samples[language].append(sentence)
                            seen_global.add(key)
                if all(len(samples[language]) >= per_language for language in ISO3_TO_ISO2.values()):
                    break
            if all(len(samples[language]) >= per_language for language in ISO3_TO_ISO2.values()):
                break

    counts = {language: len(rows) for language, rows in samples.items()}
    assert all(counts.get(language, 0) >= per_language for language in ISO3_TO_ISO2.values()), counts
    result = pd.DataFrame(
        [(sentence, language) for language in ISO3_TO_ISO2.values() for sentence in samples[language][:per_language]],
        columns=["text", "language"],
    )
    result.to_csv(output_path, index=False)
    metadata = {
        "repository": TATOEBA_REPOSITORY,
        "revision": TATOEBA_REVISION,
        "source_codes": list(ISO3_TO_ISO2),
        "target_codes": list(ISO3_TO_ISO2.values()),
        "per_language": per_language,
        "rows": len(result),
        "duplicates_after_casefold": int(result["text"].str.casefold().duplicated().sum()),
    }
    (LANGUAGE_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metadata)


if __name__ == "__main__":
    prepare_multi_news()
    prepare_tatoeba()
