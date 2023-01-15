import json
from typing import List, Optional, Dict, Any

import re
from itertools import groupby
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


class ModelOutput:
    """This class represents a piece of text generated by a language model, as well as corresponding attribute scores"""

    TEXT_REPR_MAX_LEN = 50

    def __init__(self, text: str, scores: Dict[str, float]):
        """
        :param text: the generated text
        :param scores: the attribute scores
        """
        self.text = text
        self.scores = scores

    def __repr__(self) -> str:
        text_shortcut = self.text.replace('\n', ' ')
        if len(text_shortcut) > ModelOutput.TEXT_REPR_MAX_LEN:
            text_shortcut = text_shortcut[:ModelOutput.TEXT_REPR_MAX_LEN] + '...'
        return f'Example(text="{text_shortcut}", scores={self.scores})'

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this output"""
        return {'text': self.text, 'scores': self.scores}


class Prompt:
    """This class represents a prompt, optionally with a list of corresponding model outputs"""

    def __init__(self, text: str, challenging: bool, continuations: Optional[List[ModelOutput]] = None):
        """
        :param text: the actual prompt text
        :param challenging: a flag indicating whether this prompt is considered challenging
        :param continuations: an optional list of continuations
        """
        self.text = text
        self.challenging = challenging
        self.continuations = continuations if continuations is not None else []

    def __repr__(self) -> str:
        return f'Prompt(text="{self.text}", challenging={self.challenging}, continuations=[{len(self.continuations)} Continuations])'

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this prompt"""
        return {'prompt': self.text, 'challenging': self.challenging, 'continuations': [cont.to_dict() for cont in self.continuations]}


def load_model_outputs(filename: str) -> List[ModelOutput]:
    """Load model outputs from a jsonl file in the RealToxicityPrompts format"""
    print(f'Loading model outputs from file "{filename}"')
    examples = []
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)

            if "generations" in line_json:
                line_examples = line_json["generations"]
            else:
                line_examples = [line_json]

            for ex_json in line_examples:
                text = ex_json['text']
                del ex_json['text']
                example = ModelOutput(text=text, scores=ex_json)
                examples.append(example)

    filtered_examples = [example for example in examples if not any(score is None for score in example.scores.values())]
    print(f'Done loading {len(filtered_examples)} ({len(examples)} before filtering) examples from file "{filename}"')
    return filtered_examples


def load_prompts(filename: str, challenging_only: bool = False) -> List[Prompt]:
    """Load prompts from a jsonl file in the RealToxicityPrompts format"""
    print(f'Loading prompts from file "{filename}"')
    prompts = []
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)
            prompt = Prompt(text=line_json['prompt']['text'], challenging=line_json['challenging'])
            if not challenging_only or prompt.challenging:
                prompts.append(prompt)
    print(f'Done loading {len(prompts)} {"challenging " if challenging_only else ""}prompts from file "{filename}"')
    return prompts


def mask_toxic_words(text, toxic_words, model = 'bart', already_masked=False):
    if not already_masked:
        pattern = '|'.join(toxic_words)
        word_tokenizer = TreebankWordTokenizer()
        word_detokenizer = TreebankWordDetokenizer()
        sentences = sent_tokenize(text)
        masked_sentences = []
        for sentence in sentences:
            words = word_tokenizer.tokenize(sentence)
            words = [('<mask>' if re.search(pattern, word, re.IGNORECASE) else word) for word in words]
            words = [list(group) if key!='<mask>' else list(group)[:1] for key, group in groupby(words)]
            words = [item for sublist in words for item in sublist]
            masked_sentence = word_detokenizer.detokenize(words)
            masked_sentences.append(masked_sentence)

        masked_text = ' '.join(masked_sentences)
        masked_text = re.sub("\s<mask>", f'<mask>', masked_text)
    else:
        masked_text = text
    
    if model == 't5':
        for i in range(len(re.findall("<mask>", masked_text))):
            masked_text = re.sub("<mask>", f'<extra_id_{i}>', masked_text, 1)

    return masked_text


def load_toxic_words(paths):
    toxic_words = []
    for path in paths:
        with open(path, 'r', encoding='utf8') as fh:
            for line in fh:
                toxic_words.append(line.rstrip())
    toxic_words = sorted(list(set(toxic_words)))
    return toxic_words


def token_to_word_mask(original_text, modified_text, replacement_token):
    pattern = replacement_token
    word_tokenizer = TreebankWordTokenizer()
    word_detokenizer = TreebankWordDetokenizer()

    modified_sentences = sent_tokenize(modified_text)
    original_sentences = sent_tokenize(original_text)
    try:
        assert len(modified_sentences)==len(original_sentences)
    except:
        return original_text, []

    masked_sentences = []
    masked_words = []
    for modified, original in zip(modified_sentences, original_sentences):
        modified_words = word_tokenizer.tokenize(modified)
        original_words = word_tokenizer.tokenize(original)
        modified_words = [('<mask>' if re.search(pattern, word) else word) for word in modified_words]
        masked_word_indices = [i for i, word in enumerate(modified_words) if word=='<mask>']
        try:
            # CHECK: word/token mis-alignment can happen here
            # assert len(modified_words)==len(original_words)
            masked_words += [original_words[i] for i in masked_word_indices]
        except:
            pass
        modified_words = [list(group) if key!='<mask>' else list(group)[:1] for key, group in groupby(modified_words)]
        modified_words = [item for sublist in modified_words for item in sublist]
        masked_sentence = word_detokenizer.detokenize(modified_words)
        masked_sentences.append(masked_sentence)

    masked_text = ' '.join(masked_sentences)
    masked_text = re.sub("\s<mask>", '<mask>', masked_text)
    if masked_text[-1]=='"' and original_text[-1]!='"':
        masked_text = masked_text[:-1]

    return masked_text, masked_words
