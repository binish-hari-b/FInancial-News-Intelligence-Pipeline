# Financial News Intelligence Pipeline

## Overview

Financial News Intelligence Pipeline is an end-to-end Natural Language Processing system that transforms unstructured financial news articles into structured business intelligence. The pipeline automatically extracts named entities, detects corporate events, performs financial sentiment analysis, aggregates company-level information, and generates analytical reports from a collection of financial news articles.

The project demonstrates how multiple pretrained language models can be integrated into a unified workflow for processing and analyzing financial news.

---

## Features

The pipeline provides the following functionality.

* Batch processing of financial news articles stored as text files.
* Named Entity Recognition using GLiNER.
* Zero-shot corporate event classification using ModernBERT.
* Financial sentiment analysis using FinBERT.
* Company-level information aggregation.
* Automatic analytics report generation.
* Export of structured JSON outputs.

---

## Pipeline

```text
Financial News Articles (.txt)
            │
            ▼
Document Loading
            │
            ▼
Named Entity Recognition
            │
            ▼
Corporate Event Classification
            │
            ▼
Financial Sentiment Analysis
            │
            ▼
Company Information Aggregation
            │
            ▼
Analytics Report Generation
            │
            ▼
Export Results
```

---

## Models Used

### Named Entity Recognition

Model

```
urchade/gliner_base
```

Extracted entity categories include

* Company
* Person
* Product
* Organization
* Technology
* Country
* City
* Currency

Example

```text
Apple      → Company
OpenAI     → Company
iPhone     → Product
AI         → Technology
```

---

### Corporate Event Classification

Model

```
MoritzLaurer/ModernBERT-large-zeroshot-v2.0
```

Corporate events are identified using zero-shot classification without requiring a labeled training dataset.

Supported event categories include

```text
Earnings
Revenue Guidance
Product Launch
Technology Release
Partnership
Merger
Acquisition
Investment
IPO
Executive Appointment
Executive Departure
Restructuring
Layoffs
Expansion
Bankruptcy
Lawsuit
Government Investigation
Government Regulation
Regulatory Approval
Manufacturing
Supply Chain
Patent
Licensing
Research
Market Forecast
Macroeconomic Impact
```

Predictions below a configurable confidence threshold are discarded.

---

### Financial Sentiment Analysis

Model

```
ProsusAI/finbert
```

Each article is classified as Positive, Negative, or Neutral together with a confidence score.

---

## Company Information Database

Information extracted from every article is aggregated into a company-level knowledge base.

Each company record contains

* Total number of mentions.
* Articles in which the company appears.
* Corporate events associated with the company.
* Sentiment distribution.
* Co-mentioned companies.

Example

```json
{
    "Apple": {
        "mentions": 5,
        "articles": [
            "article1.txt",
            "article8.txt"
        ],
        "events": {
            "Lawsuit": 3,
            "Product Launch": 2
        },
        "sentiments": {
            "negative": 4,
            "positive": 1
        },
        "co_companies": {
            "OpenAI": 4,
            "Google": 2
        }
    }
}
```

---

## Analytics

The generated analytics report summarizes the processed news corpus by providing

* Most mentioned companies.
* Company event distribution.
* Overall sentiment distribution.

Example

```text
FINAL ANALYTICS REPORT

Top Mentioned Companies

OpenAI ............ 12
Apple ............. 9
Microsoft ......... 7

Company Event Distribution

Lawsuit ........... 15
Investment ........ 8
Product Launch .... 7

Sentiment Distribution

Positive .......... 18
Neutral ........... 6
Negative .......... 23
```

---

## Project Structure

```text
Financial-News-Intelligence-Pipeline/

│
├── data/
│   ├── article1.txt
│   ├── article2.txt
│   └── ...
│
├── output/
│   ├── processed_articles.json
│   ├── companies_info.json
│   └── report.txt
│
└── main.py
```

---

## Output Files

### processed_articles.json

Contains the processed representation of every news article, including

* Extracted named entities.
* Detected corporate events.
* Event confidence scores.
* Financial sentiment.
* Sentiment confidence score.

---

### companies_info.json

Contains the aggregated company database, including

* Mention counts.
* Associated articles.
* Event statistics.
* Sentiment statistics.
* Co-mentioned companies.

---

### report.txt

Contains a human-readable analytics report summarizing the processed news corpus.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/binish-hari-b/FInancial-News-Intelligence-Pipeline.git

cd Financial-News-Intelligence-Pipeline
```

Install the required dependencies.

```bash
pip install torch transformers gliner accelerate sentencepiece
```

The required pretrained models are downloaded automatically during the first execution of the pipeline.

```text
urchade/gliner_base
MoritzLaurer/ModernBERT-large-zeroshot-v2.0
ProsusAI/finbert
```

---

## Usage

Place all financial news articles inside the `data` directory.

Run the pipeline.

```bash
python main.py
```

The generated outputs are saved in the `output` directory.

---

## Technologies

Python

PyTorch

Hugging Face Transformers

GLiNER

ModernBERT

FinBERT

pathlib

collections.Counter

---

## Future Improvements

Interactive dashboard for data visualization.

Semantic search over financial news articles.

Knowledge graph construction from extracted entities.

Automatic article summarization.

Time-series trend analysis.

Relationship extraction between companies and people.

Retrieval-Augmented Generation (RAG) for question answering over the news corpus.

---

## License

This project is released under the MIT License.
