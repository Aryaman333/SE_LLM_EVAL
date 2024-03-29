import nltk
import numpy as np
import spacy

class NaturalnessHelper:
    def __init__(self): 
        self.nlp = spacy.load("en_core_web_sm")
    
    def calculate_subtree_features(self,text):
        doc = self.nlp(text)
        spacy.tokens.Token.set_extension('depth', getter=lambda token: len(list(token.ancestors)), force=True)
        total_left_subtrees = 0
        total_right_subtrees = 0
        total_left_subtree_height = 0
        total_right_subtree_height = 0
        for sent in doc.sents:
            for token in sent:
                num_left_subtrees, num_right_subtrees, avg_left_subtree_height, avg_right_subtree_height = self.subtree_feature_helper(token)
                total_left_subtrees += num_left_subtrees
                total_right_subtrees += num_right_subtrees
                total_left_subtree_height += avg_left_subtree_height * num_left_subtrees
                total_right_subtree_height += avg_right_subtree_height * num_right_subtrees
        avg_left_subtree_height_text = total_left_subtree_height / total_left_subtrees if total_left_subtrees > 0 else 0
        avg_right_subtree_height_text = total_right_subtree_height / total_right_subtrees if total_right_subtrees > 0 else 0

        return avg_left_subtree_height_text,avg_right_subtree_height_text
    
    def subtree_feature_helper(self,token):
        lefts = list(token.lefts)
        rights = list(token.rights)
        num_left_subtrees = len(lefts)
        num_right_subtrees = len(rights)
        avg_left_subtree_height = sum(t._.depth for t in lefts) / num_left_subtrees if num_left_subtrees > 0 else 0
        avg_right_subtree_height = sum(t._.depth for t in rights) / num_right_subtrees if num_right_subtrees > 0 else 0
        return num_left_subtrees, num_right_subtrees, avg_left_subtree_height, avg_right_subtree_height

    def calculate_average_tree_height(self,text_list):
        total_tree_height = 0
        num_trees = 0
        for text in text_list:
            doc = self.nlp(text)
            stack = [(sent.root, 1) for sent in doc.sents]

            while stack:
                node, depth = stack.pop()
                total_tree_height += depth
                num_trees += 1
                stack.extend((child, depth + 1) for child in node.children)
        avg_tree_height = total_tree_height / max(1, num_trees)
        return avg_tree_height
    
    def avg_sentence_length(self,text):
        total_tokens = 0
        sentences= nltk.sent_tokenize(text)
        total_tokens = sum(len(sentence.split()) for sentence in sentences)
        average_length= total_tokens / len(sentences)
        return average_length

