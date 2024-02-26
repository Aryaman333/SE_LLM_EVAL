import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from nltk.translate.meteor_score import meteor_score
import gensim.downloader as api
from nltk.corpus import stopwords
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import bert_score


class Scorers:
    
    # BLEU
    def compute_bleu(self,references, candidate):
        reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
        candidate_tokens = nltk.word_tokenize(candidate.lower())

        bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        return bleu_1, bleu_2, bleu_3, bleu_4
    
    # ROGUE
    def compute_rouge(self,reference, candidate):
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = scorer.score(reference, candidate)

        return scores

    # METEOR
    def compute_meteor(self,reference, candidate):

        reference =  nltk.word_tokenize(reference.lower())
        candidate =  nltk.word_tokenize(candidate.lower())

        score = meteor_score([reference], candidate)

        return score

    # WORD MOVERS DISTANCE    
    def compute_wmd(self,sentence_a,sentence_b,model):

        sentence_a = sentence_a.lower().split()
        sentence_b = sentence_b.lower().split()


        stop_words = stopwords.words('english')
        sentence_a = [w for w in sentence_a if w not in stop_words]
        sentence_b = [w for w in sentence_b if w not in stop_words]

        distance = model.wmdistance(sentence_a,sentence_b)

        return distance
    
    # TRANSLATION ERROR RATE
    def compute_ter(self,reference, candidate):

        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())

        substitutions = nltk.edit_distance(ref_tokens, cand_tokens)
        deletions = len(ref_tokens) - len(set(ref_tokens) & set(cand_tokens))
        insertions = len(cand_tokens) - len(set(ref_tokens) & set(cand_tokens))

        reference_length = len(ref_tokens)
        ter = (substitutions + deletions + insertions) / reference_length

        return ter

    def compute_perplexity(self,text):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer(text, return_tensors = "pt")
        loss = torch.nn.CrossEntropyLoss() 
        final_loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(final_loss)
        return ppl
    
    def Bert_Score(self, candidate, reference):
        return bert_score.score(candidate, reference,lang = "en")[2].item()
        