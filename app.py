import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoTokenizer
import requests
import time

auth_token = "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"
MaskedLM_ckpt='roberta-large'
NER_tag_list = ['ORG']

@st.cache_resource
def load_seq_tagger():
    return SequenceTagger.load('ner')

@st.cache_resource
def load_tokenizer(MaskedLM_ckpt):
    return AutoTokenizer.from_pretrained(MaskedLM_ckpt)

tagger = load_seq_tagger()
MaskedLM_tok = load_tokenizer(MaskedLM_ckpt)
headers = {"Authorization": f"Bearer {auth_token}"}

API_URL_MLM = "https://api-inference.huggingface.co/models/FacebookAI/roberta-large"
def query_mlm(payload):
	response = requests.post(API_URL_MLM, headers=headers, json=payload)
	return response.json()

API_URL_CLF = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
def query_clf(payload):
	response = requests.post(API_URL_CLF, headers=headers, json=payload)
	return response.json()

def retrieve_target(input_str):
    sentence = Sentence(input_str)
    tagger.predict(sentence)
    entities = [entity.text for entity in sentence.get_spans('ner') if entity.tag in NER_tag_list]
    return entities

def prompt_sentiment_analysis(input_str):
    input_for_clf = input_str.split('<sep>')[1].strip()
    clf_output = query_clf({
        "inputs": input_for_clf,
        })
    for attempt in range(10):
        query_clf_output = query_clf({"inputs": input_for_clf})
        if not isinstance(query_clf_output, list):
            time.sleep(5)
            continue
        else:
            return query_clf_output
    return {'error': 'Max retries reached. The model is still loading.'}

def prompt_mask_filling(input_str,tgt):
    prompt = "Given this [TGT] shares are likely to [MSK]."
    prompt = prompt.replace('[TGT]',tgt).replace('[MSK]',MaskedLM_tok.mask_token)
    input_str = input_str + '<sep>' + prompt
    for attempt in range(10):
        query_mlm_output = query_mlm({"inputs": input_str})
        if not isinstance(query_mlm_output, list):
            time.sleep(5)
            continue
        else:
            max_output = max(query_mlm_output, key=lambda x: x['score'])
            return max_output['sequence']
    return {'error': 'Max retries reached. The model is still loading.'}

def run_single_absa(input_str,tgt):
    mlm_output = prompt_mask_filling(input_str,tgt)
    clf_output = prompt_sentiment_analysis(mlm_output)
    return max(clf_output[0] , key = lambda x:x['score'])

def run_absa(input_str):
    tgt_entities = retrieve_target(input_str)
    output = {e: run_single_absa(input_str, e) for e in tgt_entities}
    return output

user_input = st.text_area("Enter Financial News for Sentiment Analysis")
button = st.button("Analyze")

if user_input and button:
    final_output = run_absa(user_input)
    for Entity_Name in final_output:
        st.write("Company : ",Entity_Name,", Sentiment : ",final_output[Entity_Name]['label'].upper(),", Score : ",final_output[Entity_Name]['score'])