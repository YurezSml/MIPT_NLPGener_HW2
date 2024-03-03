import torch

import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

import bs4
import requests
from typing import List

import nltk
from nltk import sent_tokenize

from tqdm import tqdm

import numpy as np

import faiss

import re

import unicodedata

import gradio as gr
import asyncio

device = "cuda" if torch.cuda.is_available() else "cpu"
device

base_model_id = "microsoft/phi-2"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map='auto',
    trust_remote_code=True
)

ft_model = PeftModel.from_pretrained(model, "yurezsml/phi2_chan", offload_dir="./")

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text):
    text = text.lower()
    temp = remove_accents(text)
    text = text.replace('\xa0', ' ')
    text = text.replace('\n\n', '\n')
    text = text.replace('()', '')
    text = text.replace('[]', '')
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    text = text.replace('а́', 'а')
    return text

def split_text(text: str, n=2, character=" ") -> List[str]:
    text = preprocess(text)

    all_sentences = sent_tokenize(text)
    return [' '.join(all_sentences[i : i + n]) for i in range(0, len(all_sentences), 2)]


def split_documents(documents: List[str]) -> list:
    texts = []
    for text in documents:
        if text is not None:
            for passage in split_text(text):
                texts.append(passage)

    return texts


def embed(text, model, tokenizer):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
    
response = requests.get("https://en.wikipedia.org/wiki/Chandler_Bing")

base_text = ''

if response:
    html = bs4.BeautifulSoup(response.text, 'html.parser')

    title = html.select("#firstHeading")[0].text
    paragraphs = html.select("p")
    for para in paragraphs:
        base_text = base_text + para.text

fact_coh_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-multilingual-cased-sentence")
fact_coh_model = AutoModel.from_pretrained("DeepPavlov/bert-base-multilingual-cased-sentence")
fact_coh_model.to(device)

nltk.download('punkt')
subsample_documents = split_documents([base_text])

batch_size = 8
total_batches = len(subsample_documents) // batch_size + (0 if len(subsample_documents) % batch_size == 0 else 1)

base = list()
for i in tqdm(range(0, len(subsample_documents), batch_size), total=total_batches, desc="Processing Batches"):
  batch_texts = subsample_documents[i:i + batch_size]
  base.extend(embed(batch_texts, fact_coh_model, fact_coh_tokenizer))

base = np.array([vector.cpu().numpy() for vector in base])

index = faiss.IndexFlatL2(base.shape[1])
index.add(base)

async def get_context(subsample_documents, query, index, model, tokenizer):
  k = 5
  xq = embed(query.lower(), model, tokenizer).cpu().numpy()
  D, I = index.search(xq.reshape(1, 768), k)
  return subsample_documents[I[0][0]]

async def get_prompt(question, use_rag, answers_history: list[str]):
  eval_prompt = '###system: answer the question as Chandler. '
  for idx, text in enumerate(answers_history):
    if idx % 2 == 0:
      eval_prompt = eval_prompt + f' ###question: {text}'
    else:
      eval_prompt = eval_prompt + f' ###answer: {text} '
  if use_rag:
    context = await asyncio.wait_for(get_context(subsample_documents, question, index, fact_coh_model, fact_coh_tokenizer), timeout=60)
    eval_prompt = eval_prompt + f' Chandler. {context}'
  eval_prompt = eval_prompt + f' ###question: {question} '
  eval_prompt = ' '.join(eval_prompt.split())
  return eval_prompt
  
async def get_answer(question, use_rag, answers_history: list[str]):
  eval_prompt = await asyncio.wait_for(get_prompt(question, use_rag, answers_history), timeout=60)
  model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
  ft_model.eval()
  with torch.no_grad():
    answer = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=30, repetition_penalty=1.11)[0], skip_special_tokens=True) + '\n'
  answer = ' '.join(answer.split())
  if eval_prompt in answer:
    answer = answer.replace(eval_prompt,'')
  answer = answer.split('###answer')[1]

  dialog = ''
  for idx, text in enumerate(answers_history):
    if idx % 2 == 0:
      dialog = dialog + f'you: {text}\n'
    else:
      dialog = dialog + f'Chandler: {text}\n'
  dialog = dialog + f'you: {question}\n'
  dialog = dialog + f'Chandler: {answer}\n'

  answers_history.append(question)
  answers_history.append(answer)

  return dialog, answers_history
  
async def async_proc(question, use_rag, answers_history: list[str]):
  try:
    return await asyncio.wait_for(get_answer(question, use_rag, answers_history), timeout=60)
  except asyncio.TimeoutError:
    return "Processing timed out.", answers_history

gr.Interface(
    fn=async_proc,
    inputs=[
        gr.Textbox(
            label="Question",
        ),
        gr.Checkbox(label="Use RAG", info="Pick to RAG to improve factual coherence"),
        gr.State(value=[]),
    ],
    outputs=[
        gr.Textbox(
            label="Chat"
        ),
        gr.State(),
    ],
    title="Асинхронный сервис для чат-бота по сериалу Друзья",
    concurrency_limit=5
).queue().launch(share=True, debug=True)