"""Generate Assignment_3_NLP.ipynb from readable cell sources."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(
    r"""
    # Assignment №3 — Natural Language Processing

    **Выполнил:** Maidangali Backhtiyar  
    **Дата:** 23 октября 2026 г.

    ## 1. Введение

    Цель работы — реализовать и экспериментально сравнить методы NLP для анализа
    тональности, многодокументного суммирования, классификации музыкальных жанров и
    идентификации языка. Все таблицы метрик, матрицы ошибок и примеры ошибок ниже
    формируются из реальных предсказаний. Тестовые данные не используются при выборе
    гиперпараметров.

    Датасеты: `cardiffnlp/tweet_eval`, Multi-News, 
    `juliensimon/autonlp-data-song-lyrics` и `ymoslem/Tatoeba-Translations`.
    Зафиксировано случайное зерно 42.
    """
)

md("## 2. Настройка окружения\n\n### 2.1 Импорты библиотек")
code(
    r"""
    import json
    import os
    # BART generation uses a small operation not implemented by every MPS build.
    # PyTorch will execute only unsupported MPS operations on CPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import random
    import re
    import sys
    import urllib.request
    import warnings
    from collections import Counter
    from importlib.metadata import version
    from pathlib import Path

    import evaluate
    import fasttext
    import matplotlib.pyplot as plt
    import nltk
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from datasets import concatenate_datasets, load_dataset
    from huggingface_hub import hf_hub_download
    from IPython.display import Markdown, display
    from nltk.corpus import stopwords, wordnet
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import normalize
    from transformers import pipeline

    warnings.filterwarnings("ignore")
    pd.set_option("display.max_colwidth", 180)
    sns.set_theme(style="whitegrid", palette="colorblind")
    """
)

md("### 2.2 Случайное зерно и ресурсы NLTK")
code(
    r"""
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
        nltk.download(resource, quiet=True)

    EN_STOPWORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    LABEL_NAMES = ["Negative", "Neutral", "Positive"]
    """
)

md("### 2.3 Конфигурация и версии")
code(
    r"""
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "NLP_Assignment_3").exists():
        PROJECT_ROOT = PROJECT_ROOT / "NLP_Assignment_3"
    DATA_DIR = PROJECT_ROOT / "data"

    runtime_device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    device_name = "CUDA" if runtime_device == 0 else ("Apple MPS" if runtime_device == "mps" else "CPU")
    packages = ["numpy", "pandas", "scikit-learn", "nltk", "datasets", "transformers", "torch", "evaluate"]
    environment = pd.DataFrame({"Package": packages, "Version": [version(name) for name in packages]})
    print(f"Python: {sys.version.split()[0]} | Device: {device_name} | RANDOM_STATE={RANDOM_STATE}")
    display(environment)
    """
)

md(
    r"""
    ## 3. Задание 1 — Анализ тональности

    ### 3.1 Цель

    Сравниваются VADER, TF-IDF + Logistic Regression и предварительно обученная
    Transformer-модель. Исследовательский вопрос: даёт ли Transformer существенный
    прирост Macro F1 относительно словарного и традиционного подходов?

    ### 3.2 Датасет

    TweetEval Sentiment содержит метки `negative`, `neutral`, `positive`. Исходные
    разбиения объединяются, дубликаты удаляются, затем создаётся новая
    стратифицированная схема 70/15/15. Для практичного запуска берётся
    воспроизводимая сбалансированная выборка 1500 текстов каждого класса (4500 всего),
    что превышает минимум задания.
    """
)
code(
    r"""
    tweet_data = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    print("Splits:", tweet_data)
    print("Columns:", tweet_data["train"].column_names)
    print("Features:", tweet_data["train"].features)
    display(pd.DataFrame(tweet_data["train"][:3]))

    tweet_all = concatenate_datasets([tweet_data[split] for split in tweet_data.keys()]).to_pandas()
    tweet_all["text_key"] = tweet_all["text"].str.strip().str.casefold()
    duplicates_t1 = int(tweet_all["text_key"].duplicated().sum())
    tweet_all = tweet_all.drop_duplicates("text_key").drop(columns="text_key")
    raw_distribution_t1 = tweet_all["label"].value_counts().sort_index()
    print(f"После объединения: {len(tweet_all):,}; удалено дубликатов: {duplicates_t1:,}")
    display(raw_distribution_t1.rename(index=dict(enumerate(LABEL_NAMES))).to_frame("count"))

    sentiment_df = (
        tweet_all.groupby("label", group_keys=False)
        .sample(n=1500, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    train_t1, temp_t1 = train_test_split(
        sentiment_df, test_size=0.30, random_state=RANDOM_STATE, stratify=sentiment_df["label"]
    )
    valid_t1, test_t1 = train_test_split(
        temp_t1, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_t1["label"]
    )
    assert not (set(train_t1.text) & set(valid_t1.text) | set(train_t1.text) & set(test_t1.text) | set(valid_t1.text) & set(test_t1.text))
    print({"total": len(sentiment_df), "train": len(train_t1), "validation": len(valid_t1), "test": len(test_t1)})

    ax = sentiment_df["label"].map(dict(enumerate(LABEL_NAMES))).value_counts().reindex(LABEL_NAMES).plot.bar()
    ax.set(title="Распределение классов TweetEval", xlabel="Тональность", ylabel="Количество текстов")
    plt.xticks(rotation=0); plt.tight_layout(); plt.show()
    """
)

md(
    r"""
    ### 3.3 Предварительная обработка

    Для TF-IDF применяются нижний регистр, токенизация, удаление ссылок, упоминаний,
    пунктуации и стоп-слов, затем лемматизация. VADER получает почти исходный текст,
    поскольку регистр, восклицания и отрицания входят в его эвристику. Transformer
    получает исходный текст через собственный токенизатор. Чрезмерная очистка может
    уничтожить `not`, эмодзи и усилители эмоциональности.
    """
)
code(
    r"""
    def normalize_english(text):
        text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", str(text).lower())
        tokens = [token for token in word_tokenize(text) if token.isalpha()]
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in EN_STOPWORDS or token == "not"]
        return " ".join(tokens)

    preprocessing_demo_t1 = train_t1.head(5)[["text"]].copy()
    preprocessing_demo_t1["processed"] = preprocessing_demo_t1["text"].map(normalize_english)
    display(preprocessing_demo_t1)
    """
)

md("### 3.4 VADER")
code(
    r"""
    vader = SentimentIntensityAnalyzer()

    def vader_predict(texts, threshold=0.05):
        scores = np.array([vader.polarity_scores(text)["compound"] for text in texts])
        predictions = np.where(scores > threshold, 2, np.where(scores < -threshold, 0, 1))
        return predictions, scores

    vader_validation = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        predictions, _ = vader_predict(valid_t1["text"], threshold)
        vader_validation.append({"threshold": threshold, "macro_f1": f1_score(valid_t1.label, predictions, average="macro")})
    vader_validation = pd.DataFrame(vader_validation)
    best_vader_threshold = float(vader_validation.loc[vader_validation.macro_f1.idxmax(), "threshold"])
    display(vader_validation)
    print("Выбранный порог:", best_vader_threshold)
    vader_pred_t1, vader_scores_t1 = vader_predict(test_t1["text"], best_vader_threshold)
    """
)

md("### 3.5 TF-IDF + Logistic Regression")
code(
    r"""
    def make_sentiment_lr(c_value):
        return Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize_english, lowercase=False,
                                      ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)),
            ("model", LogisticRegression(C=c_value, max_iter=1000, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ])

    lr_validation = []
    for c_value in [0.1, 1.0, 10.0]:
        candidate = make_sentiment_lr(c_value)
        candidate.fit(train_t1.text, train_t1.label)
        predictions = candidate.predict(valid_t1.text)
        lr_validation.append({"C": c_value, "macro_f1": f1_score(valid_t1.label, predictions, average="macro")})
    lr_validation = pd.DataFrame(lr_validation)
    best_c_t1 = float(lr_validation.loc[lr_validation.macro_f1.idxmax(), "C"])
    display(lr_validation)
    print("Лучшее C:", best_c_t1)

    lr_sentiment = make_sentiment_lr(best_c_t1)
    lr_sentiment.fit(pd.concat([train_t1.text, valid_t1.text]), pd.concat([train_t1.label, valid_t1.label]))
    lr_pred_t1 = lr_sentiment.predict(test_t1.text)
    """
)

md(
    r"""
    ### 3.6 Transformer

    Используется `cardiffnlp/twitter-roberta-base-sentiment-latest` без дообучения.
    Инференс выполняется пакетами; входы обрезаются до 512 токенов.
    """
)
code(
    r"""
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    transformer_sentiment = pipeline(
        "text-classification", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL,
        device=runtime_device
    )
    transformer_output_t1 = transformer_sentiment(
        test_t1.text.tolist(), batch_size=32, truncation=True, max_length=512
    )

    def sentiment_label_to_id(label):
        label = label.lower()
        if "negative" in label or label.endswith("0"):
            return 0
        if "neutral" in label or label.endswith("1"):
            return 1
        if "positive" in label or label.endswith("2"):
            return 2
        raise ValueError(f"Неизвестная метка Transformer: {label}")

    transformer_pred_t1 = np.array([sentiment_label_to_id(item["label"]) for item in transformer_output_t1])
    display(pd.DataFrame(transformer_output_t1).head())
    """
)

md("### 3.7 Оценка и сравнение")
code(
    r"""
    def classification_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    predictions_t1 = {
        "VADER": vader_pred_t1,
        "TF-IDF + Logistic Regression": lr_pred_t1,
        "Transformer": transformer_pred_t1,
    }
    results_t1 = pd.DataFrame([
        {"Model": model, **classification_metrics(test_t1.label, prediction)}
        for model, prediction in predictions_t1.items()
    ]).sort_values("F1 Macro", ascending=False)
    display(results_t1.style.format({column: "{:.3f}" for column in results_t1.columns[1:]}))

    results_t1.set_index("Model")[["Accuracy", "F1 Macro"]].plot.bar(figsize=(9, 4), ylim=(0, 1))
    plt.title("Сравнение моделей тональности"); plt.ylabel("Значение метрики"); plt.xticks(rotation=15); plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (model, prediction) in zip(axes, predictions_t1.items()):
        matrix = confusion_matrix(test_t1.label, prediction, labels=[0, 1, 2])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        axis.set(title=model, xlabel="Предсказание", ylabel="Истинный класс")
    plt.tight_layout(); plt.show()
    """
)

md("### 3.8 Анализ ошибок и вывод")
code(
    r"""
    errors_t1 = test_t1[["text", "label"]].copy()
    errors_t1["VADER"] = vader_pred_t1
    errors_t1["Logistic Regression"] = lr_pred_t1
    errors_t1["Transformer"] = transformer_pred_t1
    errors_t1 = errors_t1[errors_t1["Transformer"] != errors_t1["label"]].head(5).copy()
    for column in ["label", "VADER", "Logistic Regression", "Transformer"]:
        errors_t1[column] = errors_t1[column].map(dict(enumerate(LABEL_NAMES)))
    display(errors_t1)

    best_t1 = results_t1.iloc[0]
    transformer_f1_t1 = float(results_t1.loc[results_t1.Model == "Transformer", "F1 Macro"].iloc[0])
    lr_f1_t1 = float(results_t1.loc[results_t1.Model == "TF-IDF + Logistic Regression", "F1 Macro"].iloc[0])
    display(Markdown(
        f"**Вывод.** Лучший подход — **{best_t1['Model']}** (Macro F1 = {best_t1['F1 Macro']:.3f}). "
        f"Разница Transformer и TF-IDF + Logistic Regression равна {transformer_f1_t1-lr_f1_t1:+.3f}. "
        "Ошибки в показанных примерах связаны с контекстом, неоднозначностью, сарказмом и неявной эмоциональной оценкой; "
        "поэтому вывод об улучшении делается по измеренной разнице, а не по ожиданию от архитектуры."
    ))
    """
)

md(
    r"""
    ## 4. Задание 2 — Автоматическое суммирование

    ### 4.1 Цель и выбор данных

    Сравниваются экстрактивное ранжирование предложений по TF-IDF и иерархическое
    абстрактивное суммирование BART. Репозиторий `tau/multi_news` на момент проверки
    возвращал из устаревшей Google Drive ссылки данные, не декодируемые как UTF-8.
    Поэтому прозрачно использовано зеркало исходного Multi-News
    `mehdielg/multi_news`. Зафиксирован пример `train[32766]`: ровно пять связанных
    статей, каждая длиннее 250 слов, эталон — 147 слов. Тексты сохранены в `data/` и
    не выбираются заново при каждом запуске.
    """
)
code(
    r"""
    summary_dir = DATA_DIR / "summarization"
    with open(summary_dir / "metadata.json", encoding="utf-8") as file:
        summary_metadata = json.load(file)
    selected_articles = [
        (summary_dir / f"article_{index}.txt").read_text(encoding="utf-8").strip()
        for index in range(1, 6)
    ]
    reference_summary = (summary_dir / "reference_summary.txt").read_text(encoding="utf-8").strip()
    document_info = pd.DataFrame({
        "Document": [f"Article {index}" for index in range(1, 6)],
        "Word Count": [len(article.split()) for article in selected_articles],
        "Topic": ["Назначение шерифа Дэвида Кларка в DHS"] * 5,
        "Source": ["Multi-News (mehdielg mirror), train[32766]"] * 5,
    })
    display(pd.Series(summary_metadata, name="value"))
    display(document_info)
    print(f"Reference summary ({len(reference_summary.split())} words):\n{reference_summary}")
    """
)

md("### 4.2 Предварительная обработка")
code(
    r"""
    def clean_document(text):
        return re.sub(r"\s+", " ", text.replace("NEWLINE_CHAR", " ")).strip()

    article_sentences = [sent_tokenize(clean_document(article)) for article in selected_articles]
    print("Количество предложений:", [len(sentences) for sentences in article_sentences])
    print("Пример первого предложения:", article_sentences[0][0])
    """
)

md(
    r"""
    ### 4.3 Экстрактивное суммирование

    Предложение оценивается косинусной близостью его TF-IDF-вектора к центроиду
    документа. Сначала выбираются три предложения из каждой статьи. Затем 15
    кандидатов повторно ранжируются относительно общего центроида; почти дублирующие
    предложения с косинусной близостью выше 0.75 исключаются. Финальные шесть
    предложений возвращаются в исходный порядок документов.
    """
)
code(
    r"""
    def centroid_sentence_scores(sentences):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(sentences)
        centroid = normalize(np.asarray(matrix.mean(axis=0)))
        scores = np.asarray(matrix @ centroid.T).ravel()
        return scores, matrix

    candidates = []
    per_article_extracts = []
    for document_index, sentences in enumerate(article_sentences):
        scores, _ = centroid_sentence_scores(sentences)
        chosen_indices = sorted(np.argsort(scores)[-3:])
        chosen = []
        for sentence_index in chosen_indices:
            row = {"document": document_index + 1, "sentence_index": sentence_index,
                   "sentence": sentences[sentence_index], "local_score": scores[sentence_index]}
            candidates.append(row); chosen.append(row)
        per_article_extracts.append(chosen)

    per_article_table = pd.DataFrame(candidates)
    display(per_article_table)

    candidate_texts = [row["sentence"] for row in candidates]
    global_scores, candidate_matrix = centroid_sentence_scores(candidate_texts)
    selected_candidate_indices = []
    for index in np.argsort(global_scores)[::-1]:
        if not selected_candidate_indices:
            selected_candidate_indices.append(index)
        else:
            similarities = (candidate_matrix[index] @ candidate_matrix[selected_candidate_indices].T).toarray().ravel()
            if similarities.max(initial=0) < 0.75:
                selected_candidate_indices.append(index)
        if len(selected_candidate_indices) == 6:
            break
    selected_rows = [candidates[index] for index in selected_candidate_indices]
    selected_rows.sort(key=lambda row: (row["document"], row["sentence_index"]))
    extractive_summary = " ".join(row["sentence"] for row in selected_rows)
    print(f"Extractive summary ({len(extractive_summary.split())} words, 6 sentences):\n{extractive_summary}")
    """
)

md(
    r"""
    ### 4.4 Абстрактивное суммирование

    Модель `facebook/bart-large-cnn` используется без дообучения. Каждый документ
    делится токенизатором на фрагменты до 900 токенов; каждый фрагмент суммируется,
    затем промежуточные резюме иерархически объединяются. Параметры: 4 луча,
    `do_sample=False`; для финального шага целевой диапазон 110–180 токенов с
    автоматической повторной генерацией, если результат не попал в 100–150 слов.
    """
)
code(
    r"""
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL,
                          tokenizer=SUMMARIZATION_MODEL, device=runtime_device)
    summarizer_tokenizer = summarizer.tokenizer

    def token_chunks(text, chunk_size=900):
        token_ids = summarizer_tokenizer.encode(text, add_special_tokens=False)
        return [summarizer_tokenizer.decode(token_ids[start:start + chunk_size], skip_special_tokens=True)
                for start in range(0, len(token_ids), chunk_size)]

    chunk_summaries = []
    for document_index, article in enumerate(selected_articles, start=1):
        chunks = token_chunks(article)
        outputs = summarizer(chunks, max_length=130, min_length=35, num_beams=4,
                             do_sample=False, batch_size=2)
        for output in outputs:
            chunk_summaries.append(output["summary_text"])
        print(f"Article {document_index}: {len(chunks)} chunk(s)")

    hierarchy_text = " ".join(chunk_summaries)
    while len(summarizer_tokenizer.encode(hierarchy_text, add_special_tokens=False)) > 900:
        hierarchy_outputs = summarizer(token_chunks(hierarchy_text), max_length=130, min_length=40,
                                       num_beams=4, do_sample=False, batch_size=2)
        hierarchy_text = " ".join(item["summary_text"] for item in hierarchy_outputs)

    final_min, final_max = 110, 180
    for attempt in range(3):
        abstractive_summary = summarizer(
            hierarchy_text, max_length=final_max, min_length=final_min,
            num_beams=4, do_sample=False
        )[0]["summary_text"]
        word_count = len(abstractive_summary.split())
        if 100 <= word_count <= 150:
            break
        if word_count < 100:
            final_min, final_max = min(final_min + 15, 145), min(final_max + 15, 200)
        else:
            final_min, final_max = max(90, final_min - 10), max(140, final_max - 20)
    print(f"Abstractive summary ({len(abstractive_summary.split())} words):\n{abstractive_summary}")
    """
)

md("### 4.5 ROUGE-оценка")
code(
    r"""
    rouge = evaluate.load("rouge")
    rouge_rows = []
    for method, generated in {"Extractive": extractive_summary, "Abstractive": abstractive_summary}.items():
        scores = rouge.compute(predictions=[generated], references=[reference_summary], use_stemmer=True)
        rouge_rows.append({"Method": method, "ROUGE-1": scores["rouge1"],
                           "ROUGE-2": scores["rouge2"], "ROUGE-L": scores["rougeL"]})
    rouge_results = pd.DataFrame(rouge_rows)
    display(rouge_results.style.format({"ROUGE-1": "{:.3f}", "ROUGE-2": "{:.3f}", "ROUGE-L": "{:.3f}"}))
    rouge_results.set_index("Method").plot.bar(figsize=(8, 4), ylim=(0, 1))
    plt.title("ROUGE относительно одного эталона"); plt.ylabel("ROUGE F1"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
    """
)

md("### 4.6 Качественный анализ и вывод")
code(
    r"""
    source_text = " ".join(selected_articles)
    abstractive_sentences = sent_tokenize(abstractive_summary)
    support_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(
        article_sentences[0] + article_sentences[1] + article_sentences[2] + article_sentences[3] + article_sentences[4] + abstractive_sentences
    )
    source_sentence_list = sum(article_sentences, [])
    source_matrix = support_vectorizer.transform(source_sentence_list)
    generated_matrix = support_vectorizer.transform(abstractive_sentences)
    max_support = (generated_matrix @ source_matrix.T).toarray().max(axis=1)
    least_supported_sentence = abstractive_sentences[int(np.argmin(max_support))]

    reference_sentences = sent_tokenize(reference_summary)
    omission_vectorizer = TfidfVectorizer(stop_words="english").fit(reference_sentences + [abstractive_summary])
    reference_matrix = omission_vectorizer.transform(reference_sentences)
    generated_document_vector = omission_vectorizer.transform([abstractive_summary])
    reference_coverage = (reference_matrix @ generated_document_vector.T).toarray().ravel()
    omission_candidate = reference_sentences[int(np.argmin(reference_coverage))]

    display(pd.DataFrame({
        "Критерий": ["Охват", "Связность", "Избыточность", "Фактическая согласованность", "Грамматика"],
        "Экстрактивное": ["Шесть исходных предложений", "Переходы могут быть резкими", "Возможны повторы",
                            "Формулировки сохранены", "Исходная грамматика"],
        "Абстрактивное": ["Сжимает все фрагменты иерархически", "Обычно более связное", "Сильнее устраняет повторы",
                            "Требует проверки по источникам", "Генерирует цельный текст"],
    }))
    print("Фактический кандидат на добавление/изменение (минимальная лексическая опора в источниках):")
    print(least_supported_sentence, "| support=", round(float(max_support.min()), 3))
    print("Фактический кандидат на пропуск (эталонное предложение с минимальным покрытием):")
    print(omission_candidate)

    best_rouge_method = rouge_results.set_index("Method")["ROUGE-L"].idxmax()
    display(Markdown(
        f"**Вывод.** По ROUGE-L лучший результат дал **{best_rouge_method}**. Экстрактивный метод "
        "лучше сохраняет проверяемые формулировки, но хуже связывает разные статьи; BART формирует более плавный текст, "
        "однако перефразирование требует ручной проверки. Выведенные выше предложения — реальные кандидаты из результата, "
        "а не придуманные примеры; низкое лексическое сходство является сигналом, но не доказательством галлюцинации."
    ))
    """
)

md(
    r"""
    ## 5. Задание 3 — Классификация музыкальных жанров

    ### 5.1 Цель и датасет

    Исследуются Pop, Rock, Hip Hop и Heavy Metal. Репозиторий
    `juliensimon/autonlp-data-song-lyrics` заявляет английский язык и шесть жанров.
    Его готовые Arrow-файлы имеют несовместимые схемы (`indices` против
    `text/target`), поэтому используются собственные raw CSV того же репозитория —
    это не замена датасета. После удаления дубликатов берётся по 800 песен каждого
    жанра.
    """
)
code(
    r"""
    SONG_REPOSITORY = "juliensimon/autonlp-data-song-lyrics"
    song_paths = {
        split: hf_hub_download(repo_id=SONG_REPOSITORY, filename=filename, repo_type="dataset")
        for split, filename in {"train": "raw/train2.csv", "validation": "raw/val2.csv"}.items()
    }
    print("Splits:", song_paths)
    song_raw = pd.concat([pd.read_csv(path).assign(source_split=split) for split, path in song_paths.items()], ignore_index=True)
    print("Columns:", song_raw.columns.tolist()); display(song_raw.head(3))
    song_raw = song_raw.rename(columns={"Lyric": "text", "Genre0": "genre"}).dropna(subset=["text", "genre"])
    song_raw["text_key"] = song_raw.text.str.strip().str.casefold()
    duplicate_songs = int(song_raw.text_key.duplicated().sum())
    song_raw = song_raw.drop_duplicates("text_key").drop(columns="text_key")
    selected_genres = ["Pop", "Rock", "Hip Hop", "Heavy Metal"]
    genre_counts_full = song_raw[song_raw.genre.isin(selected_genres)].genre.value_counts().reindex(selected_genres)
    display(genre_counts_full.to_frame("available_unique_lyrics"))
    print("Удалено дубликатов:", duplicate_songs)

    songs = (song_raw[song_raw.genre.isin(selected_genres)]
             .groupby("genre", group_keys=False).sample(n=800, random_state=RANDOM_STATE)
             .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))
    songs["word_count"] = songs.text.str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    songs.genre.value_counts().reindex(selected_genres).plot.bar(ax=axes[0], title="Распределение жанров")
    axes[0].set(xlabel="Жанр", ylabel="Количество песен"); axes[0].tick_params(axis="x", rotation=20)
    sns.boxplot(data=songs, x="genre", y="word_count", order=selected_genres, ax=axes[1], showfliers=False)
    axes[1].set(title="Распределение длины текстов", xlabel="Жанр", ylabel="Количество слов")
    axes[1].tick_params(axis="x", rotation=20); plt.tight_layout(); plt.show()
    display(songs.groupby("genre").word_count.describe()[["mean", "50%", "max"]])
    """
)

md("### 5.2 Предварительная обработка")
code(
    r"""
    songs["processed"] = songs.text.map(normalize_english)
    display(songs[["genre", "text", "processed"]].head(4))
    print("Очистка уменьшает влияние пунктуации и служебных слов, а лемматизация объединяет формы слова; "
          "одновременно могут исчезнуть жанрово значимые стилистические маркеры и разговорные формы.")
    """
)

md("### 5.3 Частотный анализ")
code(
    r"""
    genre_counters = {
        genre: Counter(" ".join(songs.loc[songs.genre == genre, "processed"]).split())
        for genre in selected_genres
    }
    top_words_rows = []
    for genre, counter in genre_counters.items():
        total = sum(counter.values())
        for word, count in counter.most_common(10):
            top_words_rows.append({"genre": genre, "word": word, "absolute": count, "relative_per_1000": count / total * 1000})
    top_words = pd.DataFrame(top_words_rows)
    display(top_words)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, genre in zip(axes.ravel(), selected_genres):
        subset = top_words[top_words.genre == genre].sort_values("relative_per_1000")
        axis.barh(subset.word, subset.relative_per_1000)
        axis.set(title=f"Top-10: {genre}", xlabel="На 1000 токенов", ylabel="Слово")
    plt.tight_layout(); plt.show()
    """
)

md("### 5.4 Лексические категории")
code(
    r"""
    lexical_categories = {
        "Time": {"morning", "evening", "night", "day", "today", "tomorrow", "tonight", "yesterday"},
        "Emotions": {"love", "hate", "happy", "sad", "fear", "anger", "joy", "pain", "lonely"},
        "Body Parts": {"heart", "eye", "hand", "head", "face", "body", "blood", "skin", "mouth"},
    }
    category_rows = []
    for genre, counter in genre_counters.items():
        total = sum(counter.values())
        for category, words in lexical_categories.items():
            count = sum(counter[word] for word in words)
            category_rows.append({"genre": genre, "category": category, "absolute": count,
                                  "relative_per_1000": count / total * 1000})
    category_frequencies = pd.DataFrame(category_rows)
    display(category_frequencies)
    sns.catplot(data=category_frequencies, x="genre", y="relative_per_1000", hue="category",
                kind="bar", height=4.5, aspect=1.7)
    plt.title("Лексические категории по жанрам"); plt.xlabel("Жанр"); plt.ylabel("На 1000 токенов"); plt.xticks(rotation=20); plt.show()
    """
)

md("### 5.5 Редкие и характерные слова")
code(
    r"""
    rare_rows = []
    for genre, counter in genre_counters.items():
        rare = [(word, count) for word, count in counter.items() if count <= 3]
        rare_rows.append({"genre": genre, "rare_types": len(rare),
                          "examples": ", ".join(word for word, _ in sorted(rare, key=lambda item: (-item[1], item[0]))[:12])})
    rare_words = pd.DataFrame(rare_rows)
    display(rare_words)

    genre_tfidf = TfidfVectorizer(min_df=3, max_df=0.90, max_features=20000, sublinear_tf=True)
    genre_matrix = genre_tfidf.fit_transform(songs.processed)
    vocabulary = np.array(genre_tfidf.get_feature_names_out())
    characteristic_rows = []
    for genre in selected_genres:
        mean_scores = np.asarray(genre_matrix[songs.genre.to_numpy() == genre].mean(axis=0)).ravel()
        top_indices = mean_scores.argsort()[-15:][::-1]
        characteristic_rows.append({"genre": genre, "high_tfidf_words": ", ".join(vocabulary[top_indices])})
    characteristic_words = pd.DataFrame(characteristic_rows)
    display(characteristic_words)
    print("Единица TF-IDF-документа — одна песня. Редкое слово может быть опечаткой или именем и почти не повторяться; "
          "характерное слово должно быть заметно внутри жанра и относительно менее типично для остальных документов.")
    """
)

md("### 5.6 Классификация и оценка")
code(
    r"""
    train_t3, test_t3 = train_test_split(
        songs, test_size=0.20, random_state=RANDOM_STATE, stratify=songs.genre
    )
    print({"train": len(train_t3), "test": len(test_t3)})
    print("Информация об исполнителе отсутствует, поэтому artist-group split невозможен; это потенциальное ограничение утечки.")

    genre_models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("model", MultinomialNB(alpha=0.5)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }
    genre_predictions = {}
    genre_result_rows = []
    for name, model in genre_models.items():
        model.fit(train_t3.processed, train_t3.genre)
        genre_predictions[name] = model.predict(test_t3.processed)
        genre_result_rows.append({"Model": name,
                                  "Accuracy": accuracy_score(test_t3.genre, genre_predictions[name]),
                                  "Macro F1": f1_score(test_t3.genre, genre_predictions[name], average="macro")})
    genre_results = pd.DataFrame(genre_result_rows).sort_values("Macro F1", ascending=False)
    display(genre_results.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}))
    genre_results.set_index("Model").plot.bar(figsize=(8, 4), ylim=(0, 1))
    plt.title("Классификация жанров"); plt.ylabel("Метрика"); plt.xticks(rotation=10); plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, prediction) in zip(axes, genre_predictions.items()):
        matrix = confusion_matrix(test_t3.genre, prediction, labels=selected_genres)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", ax=axis,
                    xticklabels=selected_genres, yticklabels=selected_genres)
        axis.set(title=name, xlabel="Предсказание", ylabel="Истинный жанр")
        axis.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()
    """
)

md("### 5.7 Анализ ошибок и вывод")
code(
    r"""
    best_genre_name = genre_results.iloc[0].Model
    best_genre_pred = genre_predictions[best_genre_name]
    genre_errors = test_t3[["text", "genre"]].copy()
    genre_errors["prediction"] = best_genre_pred
    genre_errors = genre_errors[genre_errors.genre != genre_errors.prediction].head(5)
    genre_errors["text"] = genre_errors.text.str.slice(0, 350) + "..."
    display(genre_errors)

    genre_cm = confusion_matrix(test_t3.genre, best_genre_pred, labels=selected_genres)
    np.fill_diagonal(genre_cm, 0)
    confused_pair_index = np.unravel_index(np.argmax(genre_cm), genre_cm.shape)
    confused_pair = f"{selected_genres[confused_pair_index[0]]} → {selected_genres[confused_pair_index[1]]}"
    display(Markdown(
        f"**Вывод.** Лучший классификатор — **{best_genre_name}** с Macro F1 = "
        f"{genre_results.iloc[0]['Macro F1']:.3f}. Наиболее частое направленное смешение: **{confused_pair}**. "
        "Лексика действительно содержит жанровый сигнал, но текст не описывает ритм, инструменты, тембр и вокальную технику; "
        "поэтому жанр нельзя полностью восстановить только по словам."
    ))
    """
)

md(
    r"""
    ## 6. Задание 4 — Идентификация языка

    ### 6.1 Цель и датасет

    Tatoeba хранит пары переводов и использует трёхбуквенные коды (`kaz`, `rus`,
    `eng`, `spa`, `por`, `fra`, `deu`), а не ожидаемые двухбуквенные. Из обеих сторон
    пар извлечено по 700 уникальных предложений для семи языков, затем коды явно
    преобразованы в `kk`, `ru`, `en`, `es`, `pt`, `fr`, `de`. Точная ревизия и
    отсутствие дубликатов записаны в metadata; локальный CSV создан программой
    `prepare_data.py` непосредственно из Parquet-файлов Hugging Face.
    """
)
code(
    r"""
    language_dir = DATA_DIR / "language_id"
    language_df = pd.read_csv(language_dir / "tatoeba_7_languages.csv")
    language_labels = ["kk", "ru", "en", "es", "pt", "fr", "de"]
    with open(language_dir / "metadata.json", encoding="utf-8") as file:
        language_metadata = json.load(file)
    print("Splits: train (source repository); columns after extraction:", language_df.columns.tolist())
    display(pd.Series(language_metadata, name="value")); display(language_df.head(7))
    language_counts = language_df.language.value_counts().sort_index()
    display(language_counts.to_frame("count"))
    assert language_counts.min() >= 500 and language_df.text.str.casefold().duplicated().sum() == 0
    language_counts.plot.bar(title="Распределение языков")
    plt.xlabel("Язык"); plt.ylabel("Количество предложений"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()

    train_t4, test_t4 = train_test_split(
        language_df, test_size=0.20, random_state=RANDOM_STATE, stratify=language_df.language
    )
    print({"train": len(train_t4), "test": len(test_t4)})
    """
)

md(
    r"""
    ### 6.2 Предварительная обработка и признаки

    Интенсивная стратегия приводит к нижнему регистру и оставляет только Unicode-буквы
    и пробелы. Поэтому казахские `ә, ғ, қ, ң, ө, ұ, ү, һ, і` сохраняются. Минимальная
    стратегия только нормализует пробелы. Словные признаки моделируют лексику, а
    символьные n-граммы 1–3 — алфавит, морфемы и орфографические шаблоны, что особенно
    полезно для коротких текстов.
    """
)
code(
    r"""
    def preprocess_intensive(text):
        letters_and_spaces = [character if character.isalpha() or character.isspace() else " "
                              for character in str(text).lower()]
        return re.sub(r"\s+", " ", "".join(letters_and_spaces)).strip()

    def preprocess_minimal(text):
        return re.sub(r"\s+", " ", str(text)).strip()

    demo_t4 = language_df.sample(5, random_state=RANDOM_STATE).copy()
    demo_t4["intensive"] = demo_t4.text.map(preprocess_intensive)
    demo_t4["minimal"] = demo_t4.text.map(preprocess_minimal)
    display(demo_t4)
    print("Kazakh characters preserved:", preprocess_intensive("Ә Ғ Қ Ң Ө Ұ Ү Һ І, 2026!"))
    """
)

md("### 6.3 Восемь обучаемых конфигураций")
code(
    r"""
    preprocessing_strategies = {"Intensive": preprocess_intensive, "Minimal": preprocess_minimal}
    feature_strategies = {
        "Word": {"analyzer": "word", "ngram_range": (1, 1), "max_features": 30000},
        "Character": {"analyzer": "char", "ngram_range": (1, 3), "max_features": 50000},
    }
    classifiers = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    }
    language_models = {}
    language_predictions = {}
    language_result_rows = []
    for prep_name, prep_function in preprocessing_strategies.items():
        for feature_name, feature_kwargs in feature_strategies.items():
            for classifier_name, classifier in classifiers.items():
                configuration = f"{prep_name} | {feature_name} | {classifier_name}"
                model = Pipeline([
                    ("tfidf", TfidfVectorizer(preprocessor=prep_function, lowercase=False,
                                              min_df=2, sublinear_tf=True, **feature_kwargs)),
                    ("model", clone(classifier)),
                ])
                model.fit(train_t4.text, train_t4.language)
                prediction = model.predict(test_t4.text)
                language_models[configuration] = model
                language_predictions[configuration] = prediction
                language_result_rows.append({"Configuration": configuration,
                                             "Accuracy": accuracy_score(test_t4.language, prediction),
                                             "Macro F1": f1_score(test_t4.language, prediction, labels=language_labels,
                                                                  average="macro", zero_division=0)})
    language_results = pd.DataFrame(language_result_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    display(language_results.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}))
    language_results.set_index("Configuration")[["Accuracy", "Macro F1"]].plot.barh(figsize=(9, 6), xlim=(0, 1))
    plt.title("Восемь конфигураций идентификации языка"); plt.xlabel("Метрика"); plt.ylabel("Конфигурация"); plt.tight_layout(); plt.show()
    """
)

md("### 6.4 Предварительно обученный fastText lid.176")
code(
    r"""
    model_dir = DATA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fasttext_path = model_dir / "lid.176.ftz"
    if not fasttext_path.exists():
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz", fasttext_path
        )
    fasttext_lid = fasttext.load_model(str(fasttext_path))

    def fasttext_predict(texts):
        labels, probabilities = fasttext_lid.predict([preprocess_minimal(text).replace("\n", " ") for text in texts])
        predictions = [label[0].replace("__label__", "") for label in labels]
        confidence = [float(probability[0]) for probability in probabilities]
        return np.array(predictions), np.array(confidence)

    fasttext_pred_t4, fasttext_confidence_t4 = fasttext_predict(test_t4.text.tolist())
    language_predictions["fastText lid.176"] = fasttext_pred_t4
    fasttext_row = {
        "Configuration": "fastText lid.176",
        "Accuracy": accuracy_score(test_t4.language, fasttext_pred_t4),
        "Macro F1": f1_score(test_t4.language, fasttext_pred_t4, labels=language_labels,
                             average="macro", zero_division=0),
    }
    all_language_results = pd.concat([language_results, pd.DataFrame([fasttext_row])], ignore_index=True).sort_values("Macro F1", ascending=False)
    display(all_language_results.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}))
    """
)

md("### 6.5 Матрица ошибок лучшей обученной конфигурации")
code(
    r"""
    best_language_configuration = language_results.iloc[0].Configuration
    best_language_model = language_models[best_language_configuration]
    best_language_pred = language_predictions[best_language_configuration]
    matrix_t4 = confusion_matrix(test_t4.language, best_language_pred, labels=language_labels)
    fasttext_matrix_t4 = confusion_matrix(test_t4.language, fasttext_pred_t4, labels=language_labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title in [
        (axes[0], matrix_t4, best_language_configuration),
        (axes[1], fasttext_matrix_t4, "fastText lid.176"),
    ]:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", ax=axis,
                    xticklabels=language_labels, yticklabels=language_labels)
        axis.set(title=title, xlabel="Предсказание", ylabel="Истинный язык")
    plt.tight_layout(); plt.show()
    """
)

md("### 6.6 Устойчивость A — длина текста")
code(
    r"""
    length_words = test_t4.text.str.split().str.len()
    length_group = pd.cut(length_words, bins=[0, 3, 10, np.inf], labels=["1–3", "4–10", ">10"])
    length_rows = []
    for group in length_group.cat.categories:
        mask = length_group == group
        length_rows.append({"Length": group, "N": int(mask.sum()),
                            "Accuracy": accuracy_score(test_t4.language[mask], best_language_pred[mask]) if mask.any() else np.nan,
                            "Macro F1": f1_score(test_t4.language[mask], best_language_pred[mask], labels=language_labels,
                                                 average="macro", zero_division=0) if mask.any() else np.nan})
    length_results = pd.DataFrame(length_rows)
    display(length_results.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}))
    length_results.set_index("Length")[["Accuracy", "Macro F1"]].plot.bar(figsize=(7, 4), ylim=(0, 1))
    plt.title("Качество по длине текста"); plt.xlabel("Количество слов"); plt.ylabel("Метрика"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
    """
)

md("### 6.7 Устойчивость B — контролируемые ошибки")
code(
    r"""
    NOISE_ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщыэюяәғқңөұүһі"

    def add_controlled_noise(text, rng):
        characters = list(str(text))
        letter_positions = [index for index, character in enumerate(characters) if character.isalpha()]
        if not letter_positions:
            return text
        operation = rng.choice(["delete", "insert", "replace"])
        position = int(rng.choice(letter_positions))
        if operation == "delete" and len(letter_positions) > 1:
            del characters[position]
        elif operation == "insert":
            characters.insert(position, rng.choice(list(NOISE_ALPHABET)))
        else:
            characters[position] = rng.choice(list(NOISE_ALPHABET))
        return "".join(characters)

    noise_rng = random.Random(RANDOM_STATE)
    noisy_test_text = test_t4.text.map(lambda text: add_controlled_noise(text, noise_rng))
    noisy_prediction = best_language_model.predict(noisy_test_text)
    robustness_results = pd.DataFrame([
        {"Condition": "Original", "Accuracy": accuracy_score(test_t4.language, best_language_pred),
         "Macro F1": f1_score(test_t4.language, best_language_pred, labels=language_labels,
                              average="macro", zero_division=0)},
        {"Condition": "With spelling error", "Accuracy": accuracy_score(test_t4.language, noisy_prediction),
         "Macro F1": f1_score(test_t4.language, noisy_prediction, labels=language_labels,
                              average="macro", zero_division=0)},
    ])
    display(robustness_results.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}))
    robustness_results.set_index("Condition").plot.bar(figsize=(7, 4), ylim=(0, 1))
    plt.title("Влияние одной контролируемой ошибки"); plt.ylabel("Метрика"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
    display(pd.DataFrame({"original": test_t4.text.head(5), "noisy": noisy_test_text.head(5)}))
    """
)

md("### 6.8 Устойчивость C — казахско-русский code-switching")
code(
    r"""
    code_switching_sentences = [
        "Мен бүгін университетке бардым, но занятия отменили.",
        "Ертең жиналыс болады, пожалуйста, не опаздывай.",
        "Бұл кітап өте қызық, я прочитал её за вечер.",
        "Ауа райы жақсы, поэтому мы пошли гулять.",
        "Мен кофені ұнатамын, но сегодня заказал чай.",
        "Сабақ сағат тоғызда басталады, аудитория на втором этаже.",
        "Ол жаңа жұмыс тапты, и теперь очень доволен.",
        "Бүгін үйде қаламын, потому что немного устал.",
        "Досым Алматыға келді, мы встретились в центре.",
        "Маған осы ән ұнайды, особенно припев.",
        "Автобус кешігіп жатыр, придётся вызвать такси.",
        "Әжем бауырсақ пісірді, получилось очень вкусно.",
        "Мен тапсырманы аяқтадым, теперь можно отдыхать.",
        "Кешке кино көреміз, билеты уже куплены.",
        "Бұл мәселе маңызды, давайте обсудим завтра.",
        "Телефонымның қуаты аз, нужна зарядка.",
        "Ол қазақша жақсы сөйлейді, но пишет по-русски.",
        "Базардан жеміс сатып алдым, цены были высокие.",
        "Бүгін қар жауды, дороги стали скользкими.",
        "Жоба дайын болды, осталось сделать презентацию.",
        "Мен жаңа рецепт көрдім, попробую приготовить вечером.",
        "Поезд таңертең келеді, встретим гостей на вокзале.",
        "Оның пікірі дұрыс, я полностью согласен.",
        "Дәрігер демалуға кеңес берді, поэтому я взял выходной.",
        "Бала мектепке кетті, а родители остались дома.",
        "Кездесу онлайн өтеді, ссылку отправят позже.",
        "Менің ноутбугым баяу жұмыс істейді, нужен новый диск.",
        "Ағам көлік сатып алды, теперь ездит на работу сам.",
        "Біз тауға шықтық, вид оттуда был прекрасный.",
        "Емтихан қиын болды, но я ответил на все вопросы.",
    ]
    code_switch_pred = best_language_model.predict(code_switching_sentences)
    if hasattr(best_language_model.named_steps["model"], "predict_proba"):
        code_switch_conf = best_language_model.predict_proba(code_switching_sentences).max(axis=1)
    else:
        decision = best_language_model.decision_function(code_switching_sentences)
        exp_scores = np.exp(decision - decision.max(axis=1, keepdims=True))
        code_switch_conf = (exp_scores / exp_scores.sum(axis=1, keepdims=True)).max(axis=1)
    code_switch_results = pd.DataFrame({"Mixed sentence": code_switching_sentences,
                                       "Predicted language": code_switch_pred,
                                       "Confidence": code_switch_conf})
    display(code_switch_results)
    print("Accuracy не рассчитывается: у смешанного текста нет единственной истинной метки без заранее принятой политики разметки.")
    """
)

md("### 6.9 Анализ ошибок и вывод")
code(
    r"""
    language_errors = test_t4[["text", "language"]].copy()
    language_errors["prediction"] = best_language_pred
    language_errors["words"] = language_errors.text.str.split().str.len()
    language_errors = language_errors[language_errors.language != language_errors.prediction].head(5)
    display(language_errors)

    clean_f1 = robustness_results.loc[robustness_results.Condition == "Original", "Macro F1"].iloc[0]
    noisy_f1 = robustness_results.loc[robustness_results.Condition == "With spelling error", "Macro F1"].iloc[0]
    display(Markdown(
        f"**Вывод.** Лучшая обученная конфигурация — **{best_language_configuration}** "
        f"(Macro F1 = {language_results.iloc[0]['Macro F1']:.3f}). После внесения ошибок Macro F1 изменился "
        f"на {noisy_f1-clean_f1:+.3f}. Символьные n-граммы часто устойчивее на коротких строках, потому что "
        "используют части слов и характерные буквы даже при отсутствии полного словарного совпадения. Ошибки особенно "
        "вероятны на коротких интернациональных словах и между испанским/португальским; code-switching принудительно "
        "сводится моделью к одному классу, поэтому уверенность не означает одноязычность."
    ))
    """
)

md("## 7. Общий вывод")
code(
    r"""
    display(Markdown(
        f"В четырёх экспериментах лучшие решения определялись только по измеренным метрикам. "
        f"Для тональности лучший Macro F1 показал **{results_t1.iloc[0].Model}** ({results_t1.iloc[0]['F1 Macro']:.3f}); "
        f"для жанров — **{genre_results.iloc[0].Model}** ({genre_results.iloc[0]['Macro F1']:.3f}); "
        f"для языка — **{all_language_results.iloc[0].Configuration}** ({all_language_results.iloc[0]['Macro F1']:.3f}). "
        f"В суммировании лучший ROUGE-L получил **{best_rouge_method}**. Результаты показывают, что более сложная "
        "модель не гарантирует преимущество: качество зависит от соответствия домену, представления признаков, длины "
        "и шума. Матрицы ошибок и реальные ошибочные примеры дополняют агрегированные метрики и показывают направления улучшения."
    ))
    """
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (NLP Assignment)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nb["nbformat"] = 4
nb["nbformat_minor"] = 5
nbf.write(nb, ROOT / "Assignment_3_NLP.ipynb")
print(f"Wrote {ROOT / 'Assignment_3_NLP.ipynb'} with {len(cells)} cells")
