from pathlib import Path
from gliner import GLiNER
from transformers import pipeline
from collections import Counter
import json
import yaml
import re

NER_LABELS = [
    "Company",
    "Person",
    "Product",
    "Organization",
    "Country",
    "City",
    "Technology",
    "Currency"
]

EVENT_LABELS = [
    "Earnings",
    "Revenue Guidance",
    "Product Launch",
    "Technology Release",
    "Partnership",
    "Merger",
    "Acquisition",
    "Divestiture",
    "Investment",
    "Fundraising",
    "IPO",
    "Share Buyback",
    "Dividend",
    "Stock Split",
    "Executive Appointment",
    "Executive Departure",
    "Restructuring",
    "Layoffs",
    "Expansion",
    "Factory Opening",
    "Factory Closure",
    "Bankruptcy",
    "Lawsuit",
    "Government Investigation",
    "Government Regulation",
    "Regulatory Approval",
    "Contract Award",
    "Cybersecurity Incident",
    "Data Breach",
    "Supply Chain",
    "Manufacturing",
    "Recall",
    "Patent",
    "Licensing",
    "Research",
    "Market Forecast",
    "Macroeconomic Impact"
]
with open('config.yaml','r') as f:
    config=yaml.safe_load(f)
FOLDER_PATH=config['data_folder_path']
OUTPUT_FOLDER_PATH=config['output_folder_path']

#LOADING DOCUMENTS
def normalize_company_name(text):
    COMPANY_SUFFIXES = r"\b(inc|incorporated|corp|corporation|ltd|limited|co|company|plc|llc|group)\b\.?"
    text=re.sub(r"[.,]","",text)
    text=re.sub(COMPANY_SUFFIXES,"",text)
    text=re.sub(r"\s+","",text).strip()
    return text

def get_text_from_files(FOLDER_PATH):
    

    folder=Path(FOLDER_PATH)
    info=[]
    for i,file in enumerate(folder.glob('*.txt')):
        try:
            text = file.read_text(encoding="utf-8").strip()
            info.append({"index": i,"filename": file.name,"text": text}) 
        except (ValueError,OSError):
            print(f"{file.name} seems to be corrupted. Skipping it...")

    return info

def extract_entities(info,ner_model,ner_labels,event_classifier,event_labels,sentiment_model,topk=3,label_threshold=0.3):
    for content in info:
        entities=ner_model.predict_entities(content["text"],ner_labels)
        content["entities"]=[]
        s=set()
        for entity in entities:
            if (entity["text"],entity["label"]) not in s:
                s.add((entity["text"],entity["label"]))
                content["entities"].append({"text": entity["text"],"label": entity["label"]})
        gt=event_classifier(content["text"],event_labels,multi_label=True,hypothesis_template="The main event discussed in the article is {}",truncation=True,max_length=8192)
        content["event_labels"]=[]
        content["confidence_score"]=[]
        i=0
        for label,score in zip(gt["labels"],gt["scores"]):
            if i == topk:break
            if(score >= label_threshold):
                content["event_labels"].append(label)
                content["confidence_score"].append(score)
                i+=1
        result=sentiment_model(content["text"],truncation=True,max_length=512)[0]
        content["sentiment"]=result["label"]
        content["sentiment_score"]=result["score"]   

    return info

def get_company_info(info):
    companies={}
    for article in info:
        article_companies = [
            normalize_company_name(entity["text"])
            for entity in article["entities"]
            if entity["label"] == "Company"
        ]

        for company in article_companies:

            if company not in companies:

                companies[company] = {
                    "mentions": 0,
                    "articles": [],
                    "events": Counter(),
                    "sentiments": Counter(),
                    "co_companies": Counter()
                }

            companies[company]["mentions"] += 1

            companies[company]["articles"].append(article["filename"])
            companies[company]["events"].update(article["event_labels"])
            companies[company]["sentiments"][article["sentiment"]] += 1

            # Every other company in the same article
            for other in article_companies:
                if other != company:
                    companies[company]["co_companies"][other] += 1

    return companies
def get_analytics_report(company_info):
    most_mentioned_companies=Counter()
    events=Counter()
    pos_sent=0
    neg_sent=0
    neu_sent=0
    for company_name,company in company_info.items():
        most_mentioned_companies[company_name]+=company["mentions"]
        for event,cnt in company["events"].items():
            events[event]+=cnt
        for sentiment,cnt in company["sentiments"].items():
            if sentiment.lower()=="positive":
                pos_sent+=cnt
            elif sentiment.lower()=="negative":
                neg_sent+=cnt
            else:neu_sent+=cnt 


    company_lines = "\n".join(
            f"      - {company}: {count} mentions" for company, count in most_mentioned_companies.most_common()
            )
    event_lines = "\n".join(
            f"      - {event}: {count} mentions" for event, count in events.most_common()
            )
    
    output=f"""
        FINAL ANALYTICS REPORT


        The mentioned companies are:

    {company_lines}


        Overall Event Distribution:

    {event_lines}

        SENTIMENT:

        Positive: {pos_sent}
        Neutral : {neu_sent}
        Negative: {neg_sent}



"""
    return output
def save_outputs(output_folder_path,info,comp_info,report):
    
    info_path=output_folder_path+r"\processed_articles.json"
    comp_info_path=output_folder_path+r"\companies_info.json"
    report_path=output_folder_path+r"\report.txt"
    with open(info_path,mode='w',encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    with open(comp_info_path,mode='w',encoding='utf-8') as f:
        json.dump(comp_info, f, indent=4, ensure_ascii=False)
    with open(report_path,mode='w',encoding='utf-8') as f:
        f.write(report)

def main():
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
    )
    model = GLiNER.from_pretrained("urchade/gliner_base")
    sentiment_model=pipeline("text-classification", model="ProsusAI/finbert")
    info=extract_entities(get_text_from_files(FOLDER_PATH),model,NER_LABELS,classifier,EVENT_LABELS,sentiment_model)
    comp_info=get_company_info(info)
    report=get_analytics_report(comp_info)
    save_outputs(OUTPUT_FOLDER_PATH,info,comp_info,report)



if __name__=="__main__":
    main()
