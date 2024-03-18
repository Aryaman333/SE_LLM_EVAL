from lexical_diversity import lex_div as ld
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from summac.model_guardrails import NERInaccuracyPenalty
import nltk
import spacy
import json
import pandas as pd
import tfidf_matcher as tm
from tfidf_matcher import ngrams, matcher
from fuzzywuzzy import fuzz
from itertools import combinations
import collections
from sklearn.preprocessing import StandardScaler
import string
from Metrics.faithfulness_helper import FaithfulnessHelper 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util

class Readability:
    def __init__(self, text):
        self.text = text
    
    def default(self):
        return self.flesch()
    
    def flesch(self): # higher = more readable
        return textstat.flesch_reading_ease(self.text)

    def reading_time(self):
        text = self.text
        return textstat.reading_time(text)

class Formality:
    def __init__(self,text):
        self.text = text
        self.tokens = ld.tokenize(text)
        self.flm_tokens = ld.flemmatize(text)
    
    def default(self):
        return self.mtld()

    def mtld(self):
        return ld.mtld(self.flm_tokens)

    def formality_score(self):
        text = self.text
        # Tokenize and tag
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Calculate the frequencies as percentages
        total_words = len(words)
        noun_freq = (pos_counts['NN'] + pos_counts['NNS'] + pos_counts['NNP'] + pos_counts['NNPS']) / total_words * 100
        adjective_freq = (pos_counts['JJ'] + pos_counts['JJR'] + pos_counts['JJS']) / total_words * 100
        preposition_freq = pos_counts['IN'] / total_words * 100
        article_freq = (pos_counts['DT'] + pos_counts['WDT']) / total_words * 100
        pronoun_freq = (pos_counts['PRP'] + pos_counts['PRP$'] + pos_counts['WP'] + pos_counts['WP$']) / total_words * 100
        verb_freq = (pos_counts['VB'] + pos_counts['VBD'] + pos_counts['VBG'] + pos_counts['VBN'] + pos_counts['VBP'] + pos_counts['VBZ']) / total_words * 100
        adverb_freq = (pos_counts['RB'] + pos_counts['RBR'] + pos_counts['RBS']) / total_words * 100
        interjection_freq = pos_counts['UH'] / total_words * 100
        
        # Formality score formula
        F = (noun_freq + adjective_freq + preposition_freq + article_freq - pronoun_freq - verb_freq - adverb_freq - interjection_freq + 100) / 2
        
        return F

    def ttr(self):
        return ld.ttr(self.flm_tokens)

class Correctness:
    def __init__(self):
        self.model_ner = NERInaccuracyPenalty()
        self.fh = FaithfulnessHelper()

    def default(self, original_text, summary):
        return self.ner_overlap([original_text], [summary])

    def ner_overlap(self, sources, generateds):
        source_ents = [self.model_ner.extract_entities(self.fh.replace_punctuation_with_whitespace(text)) for text in sources]
        generated_ents = [self.model_ner.extract_entities(self.fh.replace_punctuation_with_whitespace(text)) for text in generateds]
        similar_generated_ents = self.fh.get_similar_entities(generated_ents)
        similar_source_ents = self.fh.get_similar_entities(source_ents)
        reduced_generated_ents = self.fh.replace_similar_entities(similar_generated_ents,generated_ents)
        reduced_source_ents = self.fh.replace_similar_entities(similar_source_ents,source_ents)
        match_count,top_source_entities = self.fh.top_entities_match(reduced_source_ents,reduced_generated_ents)
        score = match_count/len(top_source_entities) if len(top_source_entities) > 0 else 0
        if len(reduced_source_ents)==0 or len(reduced_generated_ents)==0:
            return (fuzz.token_set_ratio(sources,generateds))/100
        else:
            return score
    
class fluency:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.fh = FaithfulnessHelper()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def default(self,text):
        return self.compute_perplexity(text)

    def compute_perplexity(self,text):
        inputs = self.tokenizer(text, return_tensors = "pt")
        final_loss = self.model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(final_loss)
        return ppl.item()

class relevance:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def default(self,reference,candidate):
        return self.calculate_relevance(reference,candidate)

    def calculate_relevance(self,reference,candidate):
        reference_embedding = self.model.encode(reference, convert_to_tensor=True)
        candidate_embedding = self.model.encode(candidate, convert_to_tensor=True)
        relevance_score = util.pytorch_cos_sim(reference_embedding, candidate_embedding)
        return relevance_score.item()

    