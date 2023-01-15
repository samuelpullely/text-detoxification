import math
import re
import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import string

import torch
from transformers import (
    BartTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, 
    BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM,
    ElectraTokenizer, ElectraForMaskedLM,
    GPT2LMHeadModel, GPT2Tokenizer
)

from generation import SelfDebiasingBartForConditionalGeneration, SelfDebiasingT5ForConditionalGeneration
from io_utils import mask_toxic_words, token_to_word_mask
from attention import plot_attentions, compute_rollout_attentions

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

exceptions = ['a', 'the', 'you', 'we', 'he', 'she', 'they', 'an', 'to', 'what', 'it', 'if', 'but', 'and', 'that']
exceptions += [word.capitalize() for word in exceptions]
exceptions = []

class ModelWrapper(ABC):
    """
    This class represents a wrapper for a pretrained language model that provides some high-level functions, including zero-shot
    classification using cloze questions and the generation of texts with self-debiasing.
    """

    def __init__(self, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = None
        self._model = None
    
    @abstractmethod
    def query_model_batch(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass
    
    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """Generates a continuation for a given input text."""
        pass

    @abstractmethod
    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        """
        Generates continuations for the given input texts with self-debiasing.
        :param input_texts: the input texts to generate continuations for
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param kwargs: further arguments are passed on to the original generate function
        :return: the list of generated continuations
        """
        pass

    def get_token_probability_distribution(self, input_texts: List[str], output_choices: List[str], 
                                           pattern: str, word_masking: bool = False, p_masks: float = 0.2) -> List[List[Tuple[str, float]]]:
        """
        For a batch of input texts, returns the probability distribution over possible next tokens considering only the given list of
        output choices.
        :param input_texts: the input texts
        :param output_choices: the allowed output choices (must correspond to single tokens in the model's vocabulary)
        :return: a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
        """
        output_choice_ids = []
        kwargs = {'add_prefix_space': True} if (isinstance(self, BartWrapper) or isinstance(self, RobertaWrapper) or isinstance(self, GPT2Wrapper)) else {}
        for word in output_choices:
            tokens = self._tokenizer.tokenize(word, **kwargs)
            assert len(tokens) == 1, f"Word {word} consists of multiple tokens: {tokens}"
            assert tokens[0] not in self._tokenizer.all_special_tokens, f"Word {word} corresponds to a special token: {tokens[0]}"
            token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
            output_choice_ids.append(token_id)

        if word_masking:
            logits, masked_texts, masked_words_per_text = self.query_model_batch_dev(input_texts, pattern, p_masks)
        else:
            logits = self.query_model_batch(input_texts)
        
        result = []
        for idx, _ in enumerate(input_texts):
            output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
            choices_with_probabilities = list(zip(output_choices, (prob.item() for prob in output_probabilities)))
            result.append(choices_with_probabilities)

        if word_masking:
            return result, masked_texts, masked_words_per_text
        else:
            return result
        

class BartWrapper(ModelWrapper):

    def __init__(self, model_name: str = "facebook/bart-large", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained BART model (default: "facebook/bart-large")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = BartTokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingBartForConditionalGeneration.from_pretrained(model_name, forced_bos_token_id=0).to(self._device)
        self._model.eval()

    def query_model_batch(self, input_texts: List[str]):
        encoder_inputs = self._tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        encoder_inputs = {key: val.to(self._device) for key, val in encoder_inputs.items()}

        decoder_input_texts = [text[:-len('<mask>')] for text in input_texts]
        
        decoder_inputs = self._tokenizer(decoder_input_texts, padding=True, truncation=True, return_tensors='pt')
        shifts = decoder_inputs['attention_mask'].shape[-1] - decoder_inputs['attention_mask'].sum(dim=-1) + 1
        for batch_idx in range(decoder_inputs['input_ids'].shape[0]):
            decoder_inputs['input_ids'][batch_idx] = decoder_inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())
        
        decoder_inputs['attention_mask'] = torch.where(decoder_inputs['input_ids']==self._model.config.pad_token_id, 0, 1)
        decoder_inputs = {k: v.to(self._device) for k, v in decoder_inputs.items()}
        kwargs = {}
        kwargs['decoder_input_ids'] = decoder_inputs['input_ids']
        kwargs['decoder_attention_mask'] = decoder_inputs['attention_mask']

        input_length = decoder_inputs['input_ids'].shape[1]
        min_length = 2 + input_length
        max_length = min_length
        
        with torch.no_grad():
            output = self._model.generate(inputs=encoder_inputs['input_ids'], attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length, use_cache=False, num_beams=1, 
                                          return_dict_in_generate=True, output_scores=True, output_attentions=True, **kwargs)
        logits = output.scores[0]        
        return logits
    
    def query_model_batch_dev(self, input_texts: List[str], pattern: str):
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        encoder_inputs = self._tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        encoder_inputs = {key: val.to(self._device) for key, val in encoder_inputs.items()}

        decoder_input_texts = [text[:-len('<mask>')] for text in input_texts]
        
        decoder_inputs = self._tokenizer(decoder_input_texts, padding=True, truncation=True, return_tensors='pt')
        shifts = decoder_inputs['attention_mask'].shape[-1] - decoder_inputs['attention_mask'].sum(dim=-1) + 1
        for batch_idx in range(decoder_inputs['input_ids'].shape[0]):
            decoder_inputs['input_ids'][batch_idx] = decoder_inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())
        
        decoder_inputs['attention_mask'] = torch.where(decoder_inputs['input_ids']==self._model.config.pad_token_id, 0, 1)
        decoder_inputs = {k: v.to(self._device) for k, v in decoder_inputs.items()}
        kwargs = {}
        kwargs['decoder_input_ids'] = decoder_inputs['input_ids']
        kwargs['decoder_attention_mask'] = decoder_inputs['attention_mask']

        input_length = decoder_inputs['input_ids'].shape[1]
        min_length = 2 + input_length
        max_length = min_length

        with torch.no_grad():
            output = self._model.generate(input_ids=encoder_inputs['input_ids'], attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length, use_cache=False, num_beams=1, 
                                          return_dict_in_generate=True, output_scores=True, output_attentions=True, **kwargs)
            # type(output) --> GreedySearchEncoderDecoderOutput
        logits = output.scores[0]

        # encoder_inputs['input_ids'].shape
        # torch.Size([3, 33])
        # decoder_inputs['input_ids'].shape
        # torch.Size([3, 32])
        # output.sequences.shape
        # torch.Size([3, 34])
        input_strings = self._tokenizer.batch_decode(encoder_inputs['input_ids'])
        batch_size = encoder_inputs['input_ids'].shape[0]
        tokenized_encoder_inputs = []
        tokenized_outputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_encoder_inputs.append(self._tokenizer.convert_ids_to_tokens(encoder_inputs['input_ids'][i]))
            tokenized_outputs.append(self._tokenizer.convert_ids_to_tokens(output.sequences[i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        # ### encoder_attentions
        # # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation_utils.GreedySearchEncoderDecoderOutput
        # encoder_attentions = output['encoder_attentions']
        # raw_attentions = torch.stack(encoder_attentions, dim=1)
        # # raw_attentions.shape
        # # [3, 12, 16, 33, 33] --> [batch_size, num_layers, num_heads, encoder_input_length, encoder_input_length]
        # batch_size, num_layers, num_heads, encoder_input_length, _ = raw_attentions.shape
        # example_lengths = encoder_input_length - (encoder_inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        # masked_index = (encoder_inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)

        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     # plot all tokens
        #     tokenized_encoder_input = tokenized_encoder_inputs[batch_index]
        #     query_index = masked_index[1][batch_index]
        #     key_indices = torch.tensor(range(example_lengths[batch_index]))
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_encoder_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-encoder-raw-attentions-{batch_index}.png", dpi=500)
        #     # plot original tokens only (without pattern tokens)
        #     fig, ax = plt.subplots()
        #     original_tokenized_input = original_tokenized_inputs[batch_index]
        #     key_indices = torch.tensor(range(2, len(original_tokenized_input)))
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_encoder_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-encoder-original-raw-attentions-{batch_index}.png", dpi=500)

        # for batch_index in range(batch_size):
        #     example_length = example_lengths[batch_index]
        #     attentions = raw_attentions[batch_index, :, :, :example_length, :example_length] # attentions.shape = [num_layers, num_heads, example_length, example_length]
        #     # average over num_heads
        #     attentions = attentions.sum(dim=1) / attentions.shape[1] # attentions.shape = [num_layers, example_length, example_length]
        #     # add residual weights
        #     residual_weights = torch.eye(example_length)[None] # residual_weights.shape = [1, example_length, example_length]
        #     residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
        #     # CHECK: maybe use softmax for normalization?
        #     # normalize attention weights
        #     residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
        #     residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
        #     # compute rollout_attentions
        #     rollout_attentions = compute_rollout_attentions(residual_attentions) # rollout_attentions.shape = [num_layers, example_length, example_length]
        #     # plot residual_attentions and rollout_attentions
        #     query_index = masked_index[1][batch_index]
        #     original_tokenized_input = original_tokenized_inputs[batch_index]
        #     tokenized_encoder_input = tokenized_encoder_inputs[batch_index]
        #     all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
        #     for name, attentions in all_attentions.items():
        #         # plot all tokens
        #         fig, ax = plt.subplots()
        #         key_indices = torch.tensor(range(example_lengths[batch_index]))
        #         plot_attentions(attentions, key_indices, tokenized_encoder_input, query_index=query_index, head_average=False, ax=ax)
        #         ax.set_title(original_texts[batch_index])
        #         plt.tight_layout()
        #         fig.savefig(f"figures/{self._model_name}-encoder-{name}-{batch_index}.png", dpi=500)
        #         # plot original tokens only (without pattern tokens)
        #         fig, ax = plt.subplots()
        #         key_indices = torch.tensor(range(2, len(original_tokenized_input)))
        #         plot_attentions(attentions, key_indices, tokenized_encoder_input, query_index=query_index, head_average=False, ax=ax)
        #         ax.set_title(original_texts[batch_index])
        #         plt.tight_layout()
        #         fig.savefig(f"figures/{self._model_name}-encoder-original-{name}-{batch_index}.png", dpi=500)      
        
        # ### decoder_attentions
        # decoder_attentions = output['decoder_attentions']
        # # len(decoder_attentions)
        # # 2
        # # len(decoder_attentions[0])
        # # 12
        # # decoder_attentions[0][-1].shape
        # # torch.Size([3, 16, 32, 32])
        # # decoder_attentions[1][-1].shape
        # # torch.Size([3, 16, 33, 33])
        # raw_attentions = torch.stack(decoder_attentions[0], dim=1)
        # # raw_attentions.shape
        # # [3, 12, 16, 32, 32] --> [batch_size, num_layers, num_heads, generated_sequence_length, generated_sequence_length]
        # batch_size, num_layers, num_heads, generated_sequence_length, _ = raw_attentions.shape
        # num_pad_tokens = (decoder_inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        # example_lengths = generated_sequence_length - num_pad_tokens
        
        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     tokenized_output = tokenized_outputs[batch_index]
        #     query_index = -1
        #     # plot all tokens
        #     key_indices = torch.zeros(example_lengths[batch_index], dtype=torch.long)
        #     key_indices[1:] = torch.tensor(range(num_pad_tokens[batch_index] + 1, generated_sequence_length))
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_output, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-raw-attentions-{batch_index}.png", dpi=500)
        #     # plot original tokens only (without pattern and special tokens)
        #     original_tokenized_input = original_tokenized_inputs[batch_index]
        #     start_index = num_pad_tokens[batch_index] + 3
        #     # solves issue with text that starts with special symbols which are not alphanumeric characters
        #     # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
        #     if tokenized_output[start_index-1]=='"': 
        #         pass
        #     else:
        #         start_index = start_index - 1
        #     end_index = len(original_tokenized_input) + start_index - 2
        #     key_indices = torch.tensor(range(start_index, end_index))
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_output, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-original-raw-attentions-{batch_index}.png", dpi=500)
        
        # for batch_index in range(batch_size):
        #     example_length = example_lengths[batch_index]
        #     key_indices = torch.zeros(example_length, dtype=torch.long)
        #     key_indices[1:] = torch.tensor(range(num_pad_tokens[batch_index] + 1, generated_sequence_length))
        #     attentions = raw_attentions[batch_index, :, :, key_indices] # attentions.shape = [num_layers, num_heads, example_length, generated_sequence_length]
        #     attentions = attentions[:, :, :, key_indices] # attentions.shape = [num_layers, num_heads, example_length, example_length]
        #     # average over num_heads
        #     attentions = attentions.mean(dim=1) # attentions.shape = [num_layers, example_length, example_length]
        #     # add residual weights
        #     residual_weights = torch.eye(example_length)[None].to(self._device) # residual_weights.shape = [1, example_length, example_length]
        #     residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
        #     # CHECK: maybe use softmax for normalization?
        #     # normalize attention weights
        #     residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
        #     residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
        #     # compute rollout_attentions
        #     rollout_attentions = compute_rollout_attentions(residual_attentions)
        #     # plot residual_attentions and rollout_attentions
        #     query_index = -1
        #     tokenized_output = [token for token in tokenized_outputs[batch_index] if token != '<pad>']
        #     all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
        #     for name, weights in all_attentions.items():
        #         # plot all tokens
        #         key_indices = torch.tensor(range(example_length))
        #         fig, ax = plt.subplots()
        #         plot_attentions(weights, key_indices, tokenized_output, head_average=False, query_index=query_index, ax=ax)
        #         ax.set_title(original_texts[batch_index])
        #         plt.tight_layout()
        #         fig.savefig(f"figures/{self._model_name}-decoder-{name}-{batch_index}.png", dpi=500)
        #         # plot original tokens only (without pattern and special tokens)
        #         original_tokenized_input = original_tokenized_inputs[batch_index]
        #         start_index = 3
        #         end_index = len(original_tokenized_input) + start_index - 2
        #         key_indices = torch.tensor(range(start_index, end_index))
        #         fig, ax = plt.subplots()
        #         plot_attentions(weights, key_indices, tokenized_output, head_average=False, query_index=query_index, ax=ax)
        #         ax.set_title(original_texts[batch_index])
        #         plt.tight_layout()
        #         fig.savefig(f"figures/{self._model_name}-decoder-original-{name}-{batch_index}.png", dpi=500)
        

        ## cross_attentions
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation_utils.GreedySearchEncoderDecoderOutput
        cross_attentions = output['cross_attentions']
        # len(cross_attentions)
        # 2
        # len(cross_attentions[0])
        # 12
        # cross_attentions[0][-1].shape
        # torch.Size([3, 16, 32, 33]) --> [batch_size, num_heads, decoder_sequence_length, encoder_input_length]
        # cross_attentions[1][-1].shape
        # torch.Size([3, 16, 32+1, 33])
        raw_attentions = torch.stack(cross_attentions[0], dim=1)
        # raw_attentions.shape
        # [3, 12, 16, 32, 33] --> [batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length]
        batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length = raw_attentions.shape
        example_lengths = encoder_input_length - (encoder_inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)

        masked_texts = []
        masked_words_per_text = []
        # plot raw_attentions
        for batch_index in range(batch_size):
            # tokenized_encoder_input = tokenized_encoder_inputs[batch_index]
            # query_index = -1
            # # plot all tokens
            # key_indices = torch.tensor(range(example_lengths[batch_index]))
            # fig, ax = plt.subplots()
            # plot_attentions(raw_attentions[batch_index], key_indices, tokenized_encoder_input, query_index=query_index, ax=ax)
            # ax.set_title(original_texts[batch_index])
            # plt.tight_layout()
            # fig.savefig(f"figures/{self._model_name}-decoder-cross-raw-attentions-{batch_index}.png", dpi=500)
            # # plot original tokens only (without pattern and special tokens)
            # original_tokenized_input = original_tokenized_inputs[batch_index]
            # # solves issue with text that starts with special symbols which are not alphanumeric characters
            # # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            # if tokenized_encoder_input[1]=='"': 
            #     offset = 2
            # else:
            #     offset = 1
            # key_indices = torch.tensor(range(offset, len(original_tokenized_input)+offset-2))
            # fig, ax = plt.subplots()
            # plot_attentions(raw_attentions[batch_index], key_indices, tokenized_encoder_input, query_index=query_index, ax=ax)
            # ax.set_title(original_texts[batch_index])
            # plt.tight_layout()
            # fig.savefig(f"figures/{self._model_name}-decoder-cross-original-raw-attentions-{batch_index}.png", dpi=500)

            # find out which tokens in the original sentence need to be masked
            tokenized_encoder_input = tokenized_encoder_inputs[batch_index]
            original_tokenized_input = original_tokenized_inputs[batch_index]
            layer_index = -1
            query_index = -1
            # solves issue with text that starts with special symbols which are not alphanumeric characters
            # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            if tokenized_encoder_input[1]=='"': 
                offset = 2
            else:
                offset = 1
            key_indices = torch.tensor(range(offset, len(original_tokenized_input)+offset-2))
            attentions = raw_attentions[batch_index].mean(dim=1)
            attention_weights = attentions[layer_index, query_index, key_indices]
            num_masks = min(2, attention_weights.shape[0])
            values, indices = torch.topk(attention_weights, k=num_masks, dim=-1)
            indices = indices + offset
            # print(indices)
            print('TOP TOKENS:', [tokenized_encoder_input[token_index] for token_index in indices], values.cpu().numpy().tolist())
            # mask tokens with large attention weights
            original_text = original_texts[batch_index]
            print(f'ORIGINAL: {original_text}')
            replacement_token = 'UNIQUEREPLACEMENTTOKEN'
            tokenized_text = original_tokenized_input
            replacement_indices = indices - offset + 1
            for i in replacement_indices:
                # if not tokenized_text[i].replace('Ġ', '') in string.punctuation:
                if len(set(tokenized_text[i]).intersection(set(string.punctuation)))==0 and re.sub('|'.join(['Ġ', ' ']), '', tokenized_text[i]) not in exceptions:
                    if tokenized_text[i].startswith('Ġ'):
                        tokenized_text[i] = 'Ġ' + replacement_token
                    else:
                        tokenized_text[i] = replacement_token
            modified_text = self._tokenizer.convert_tokens_to_string(tokenized_text[1:-1])
            print(f'MODIFIED: {modified_text}')
            masked_text, masked_words = token_to_word_mask(original_text, modified_text, replacement_token)
            masked_texts.append(masked_text)
            masked_words_per_text.append(masked_words)
            print(f'MASKED: {masked_text}')
            print(f'MASKED WORDS: {masked_words}')
            print()


        return logits, masked_texts, masked_words_per_text

    def generate(self, input_text: str, **kwargs):
        input_ids = self._tokenizer(input_text, return_tensors='pt').input_ids.to(self._device)
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(self, input_texts: List[str], toxic_words: List[str], debiasing_prefixes: List[str], 
                                decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False, 
                                min_length: int = None, max_length: int = None, 
                                reg_temperature: float = 1.0, bias_temperature: float = 1.0, already_masked: bool = False, **kwargs) -> List[str]:
        
        num_debiasing_prefixes = len(debiasing_prefixes)
        self._model.init_logits_processor(num_debiasing_prefixes=num_debiasing_prefixes, decay_constant=decay_constant, 
                                          epsilon=epsilon, debug=debug, tokenizer=self._tokenizer, 
                                          reg_temperature=reg_temperature, bias_temperature=bias_temperature)
        
        input_texts = [mask_toxic_words(text, toxic_words, 'bart', already_masked=already_masked) for text in input_texts]
        encoder_inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_sentence in input_texts:
                # encoder_inputs += [debiasing_prefix + mask_toxic_words(input_sentence, toxic_words)]
                encoder_inputs += [debiasing_prefix + input_sentence]
        
        encoder_inputs = self._tokenizer(encoder_inputs, padding=True, truncation=True, return_tensors='pt')
        encoder_inputs = {k: v.to(self._device) for k, v in encoder_inputs.items()}

        n_input_texts = len(input_texts)
        decoder_inputs = [''] * n_input_texts
        for debiasing_prefix in debiasing_prefixes:
            for input_text in [''] * n_input_texts:
                decoder_inputs += [debiasing_prefix + input_text]

        decoder_inputs = self._tokenizer(decoder_inputs, padding=True, truncation=True, return_tensors='pt')
        decoder_inputs['attention_mask'] = torch.flip(decoder_inputs['attention_mask'], dims=[1])
        shifts = decoder_inputs['attention_mask'].shape[-1] - decoder_inputs['attention_mask'].sum(dim=-1) + 1
        for batch_idx in range(decoder_inputs['input_ids'].shape[0]):
            decoder_inputs['input_ids'][batch_idx] = decoder_inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        decoder_inputs['attention_mask'] = torch.where(decoder_inputs['input_ids']==self._model.config.pad_token_id, 0, 1)
        decoder_inputs = {k: v.to(self._device) for k, v in decoder_inputs.items()}
        kwargs['decoder_input_ids'] = decoder_inputs['input_ids']
        kwargs['decoder_attention_mask'] = decoder_inputs['attention_mask']

        input_length = decoder_inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(inputs=encoder_inputs['input_ids'], attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length, use_cache=False, **kwargs)

        batch_size = output_ids.shape[0] // (1 + num_debiasing_prefixes)
        output_ids = output_ids[:batch_size]
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)


class T5Wrapper(ModelWrapper):
    """A wrapper for the T5 model"""

    def __init__(self, model_name: str = "google/t5-v1_1-large", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained T5 model (default: "google/t5-v1_1-large")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingT5ForConditionalGeneration.from_pretrained(model_name)
        self._model.eval()
        if self._device == 'cuda':
            self._model.parallelize()

    def query_model_batch(self, input_texts: List[str]):
        assert all('<extra_id_0>' in input_text for input_text in input_texts)
        output_texts = ['<extra_id_0>'] * len(input_texts)
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_ids = self._tokenizer.batch_encode_plus(output_texts, return_tensors='pt')['input_ids'].to(self._device)
        with torch.no_grad():
            outputs = self._model(labels=output_ids, **inputs, output_attentions=True)
        logits = outputs['logits'][:, 1, :]        
        return logits

    def query_model_batch_dev(self, input_texts: List[str], pattern: str):
        assert all('<extra_id_0>' in input_text for input_text in input_texts)
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        output_texts = ['<extra_id_0>'] * len(input_texts)
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_ids = self._tokenizer.batch_encode_plus(output_texts, return_tensors='pt')['input_ids'].to(self._device)
        with torch.no_grad():
            outputs = self._model(labels=output_ids, **inputs, output_attentions=True)
            # type(outputs) --> Seq2SeqLMOutput
        logits = outputs['logits'][:, 1, :]
        # cross_attentions = outputs['cross_attentions']

        # inputs['input_ids'].shape
        # torch.Size([3, 36])
        # output_ids.shape
        # torch.Size([3, 2])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        # ## cross_attentions
        # # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput
        # cross_attentions = outputs['cross_attentions']
        # # len(cross_attentions)
        # # 24 = num_layers
        # # cross_attentions[-1].shape
        # # [3, 16, 2, 36] --> [batch_size, num_heads, decoder_sequence_length, encoder_input_length]
        # raw_attentions = torch.stack(cross_attentions, dim=1)
        # # raw_attentions.shape
        # # [3, 24, 16, 2, 36] --> [batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length]
        # batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length = raw_attentions.shape
        # num_pad_tokens = (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        # example_lengths = encoder_input_length - num_pad_tokens

        # masked_texts = []
        # masked_words_per_text = []
        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     # find out which tokens in the original sentence need to be masked
        #     tokenized_input = tokenized_inputs[batch_index]
        #     layer_index = -1
        #     query_index = -1
        #     # solves issue with text that starts with special symbols which are not alphanumeric characters
        #     # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
        #     if tokenized_input[0]=='▁"': 
        #         offset = 1
        #     else:
        #         offset = 0
        #     test_index = -21 - num_pad_tokens[batch_index]
        #     if tokenized_input[test_index]=='"':
        #         end_index = test_index
        #     else:
        #         end_index = test_index + 1
        #     final_index = max([i for i, _ in enumerate(tokenized_input[:end_index])]) + 1
        #     key_indices = torch.tensor(range(offset, final_index)) # print([tokenized_input[i] for i in key_indices])
        #     attentions = raw_attentions[batch_index].mean(dim=1)
        #     attention_weights = attentions[layer_index, query_index, key_indices]
        #     num_masks = min(5, attention_weights.shape[0])
        #     values, indices = torch.topk(attention_weights, k=num_masks, dim=-1)
        #     indices = indices + offset
        #     # print(indices)
        #     print('TOP TOKENS:', [tokenized_input[token_index] for token_index in indices], values.cpu().numpy().tolist())
        #     # mask tokens with large attention weights
        #     original_text = original_texts[batch_index]
        #     print(f'ORIGINAL: {original_text}')
        #     replacement_token = 'UNIQUEREPLACEMENTTOKEN'
        #     tokenized_text = tokenized_input.copy() #original_tokenized_input
        #     replacement_indices = indices# - offset
        #     for i in replacement_indices:
        #         if len(set(tokenized_text[i]).intersection(set(string.punctuation)))==0 and (re.sub('|'.join(['▁', ' ']), '', tokenized_text[i]) not in exceptions) and (tokenized_text[i]!='▁'):
        #             if tokenized_text[i].startswith('▁'):
        #                 tokenized_text[i] = ' ' + replacement_token
        #             else:
        #                 tokenized_text[i] = replacement_token
        #     modified_text = self._tokenizer.convert_tokens_to_string([tokenized_text[i] for i in key_indices])
        #     print(f'MODIFIED: {modified_text}')
        #     masked_text, masked_words = token_to_word_mask(original_text, modified_text, replacement_token)
        #     masked_texts.append(masked_text)
        #     masked_words_per_text.append(masked_words)
        #     print(f'MASKED: {masked_text}')
        #     print(f'MASKED WORDS: {masked_words}')
        #     print()
      
        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     tokenized_input = tokenized_inputs[batch_index]
        #     query_index = -1
        #     # plot all tokens
        #     key_indices = torch.tensor(range(example_lengths[batch_index])) #print([tokenized_input[i] for i in key_indices])
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-cross-raw-attentions-{batch_index}.png", dpi=500)
        #     # plot original tokens only (without pattern and special tokens)
        #     if tokenized_input[0]=='▁"': 
        #         offset = 1
        #     else:
        #         offset = 0
        #     test_index = -21 - num_pad_tokens[batch_index]
        #     if tokenized_input[test_index]=='"':
        #         end_index = test_index
        #     else:
        #         end_index = test_index + 1
        #     final_index = max([i for i, _ in enumerate(tokenized_input[:end_index])]) + 1
        #     key_indices = torch.tensor(range(offset, final_index)) # print([tokenized_input[i] for i in key_indices])
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-cross-original-raw-attentions-{batch_index}.png", dpi=500)


        ## encoder_attentions
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput
        encoder_attentions = outputs['encoder_attentions']
        # len(encoder_attentions)
        # 24 = num_layers
        # encoder_attentions[-1].shape
        # [4, 16, 36, 36] --> [batch_size, num_heads, encoder_input_length, encoder_input_length]
        raw_attentions = torch.stack(encoder_attentions, dim=1)
        # raw_attentions.shape
        # [4, 24, 16, 36, 36] --> [batch_size, num_layers, num_heads, encoder_input_length, encoder_input_length]
        batch_size, num_layers, num_heads, encoder_input_length, _ = raw_attentions.shape
        num_pad_tokens = (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        example_lengths = encoder_input_length - num_pad_tokens
        mask_token_id = 32099
        masked_index = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)

        # plot raw_attentions
        for batch_index in range(batch_size):
            tokenized_input = tokenized_inputs[batch_index]
            query_index = masked_index[1][batch_index]
            # plot all tokens
            key_indices = torch.tensor(range(example_lengths[batch_index])) #print([tokenized_input[i] for i in key_indices])
            fig, ax = plt.subplots()
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-encoder-raw-attentions-{batch_index}.png", dpi=500)
            # plot original tokens only (without pattern and special tokens)
            if tokenized_input[0]=='▁"': 
                offset = 1
            else:
                offset = 0
            test_index = -21 - num_pad_tokens[batch_index]
            if tokenized_input[test_index]=='"':
                end_index = test_index
            else:
                end_index = test_index + 1
            final_index = max([i for i, _ in enumerate(tokenized_input[:end_index])]) + 1
            key_indices = torch.tensor(range(offset, final_index)) # print([tokenized_input[i] for i in key_indices])
            fig, ax = plt.subplots()
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-encoder-original-raw-attentions-{batch_index}.png", dpi=500)



        return logits, masked_texts, masked_words_per_text

    def query_model_batch_dev_alt(self, input_texts: List[str], pattern: str):
        assert all('<extra_id_0>' in input_text for input_text in input_texts)
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        output_texts = ['<extra_id_0>'] * len(input_texts)
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        
        min_length = max_length = 3
        with torch.no_grad():
            output = self._model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length, num_beams=1, 
                                          return_dict_in_generate=True, output_scores=True, output_attentions=True)
            # type(output) --> GreedySearchEncoderDecoderOutput
        logits = output.scores[-1]
        # cross_attentions = outputs['cross_attentions']

        # # inputs['input_ids'].shape
        # # torch.Size([1, 36])
        # # output_ids.shape
        # # torch.Size([1, 2])
        # tokenized_input = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # final_input_token_index = -21
        # input_string = self._tokenizer.convert_tokens_to_string(tokenized_input)
        # tokenized_output = self._tokenizer.convert_ids_to_tokens(output_ids[0])
        # answer = self._tokenizer.convert_ids_to_tokens(logits.argmax().item())
        # extra_id_token = self._tokenizer.convert_ids_to_tokens(outputs['logits'][:, 0, :].argmax().item())
        # output_probabilities = logits[0][[2163, 465]].softmax(dim=0)

        # ### cross_attentions
        # # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput
        # # len(cross_attentions)
        # # 24 = num_layers
        # # cross_attentions[-1].shape
        # # torch.Size([1, 16, 2, 36]) --> [batch_size, num_heads, generated_sequence_length, encoder_input_length]
        # generated_token_index = 1


        # inputs['input_ids'].shape
        # torch.Size([3, 36])
        # output_ids.shape
        # torch.Size([3, 2])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        original_tokenized_inputs = []
        tokenized_outputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            tokenized_outputs.append(self._tokenizer.convert_ids_to_tokens(output.sequences[i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        ## cross_attentions
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput
        cross_attentions = output['cross_attentions'][-1]
        # len(cross_attentions)
        # 24 = num_layers
        # cross_attentions[-1].shape
        # [3, 16, 2, 36] --> [batch_size, num_heads, decoder_sequence_length, encoder_input_length]
        raw_attentions = torch.stack(cross_attentions, dim=1)
        # raw_attentions.shape
        # [3, 24, 16, 2, 36] --> [batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length]
        batch_size, num_layers, num_heads, decoder_sequence_length, encoder_input_length = raw_attentions.shape
        num_pad_tokens = (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        example_lengths = encoder_input_length - num_pad_tokens

        masked_texts = []
        masked_words_per_text = []
        # plot raw_attentions
        for batch_index in range(batch_size):
            # find out which tokens in the original sentence need to be masked
            tokenized_input = tokenized_inputs[batch_index]
            layer_index = -1
            query_index = -1
            # solves issue with text that starts with special symbols which are not alphanumeric characters
            # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            if tokenized_input[0]=='▁"': 
                offset = 1
            else:
                offset = 0
            test_index = -21 - num_pad_tokens[batch_index]
            if tokenized_input[test_index]=='"':
                end_index = test_index
            else:
                end_index = test_index + 1
            final_index = max([i for i, _ in enumerate(tokenized_input[:end_index])]) + 1
            key_indices = torch.tensor(range(offset, final_index)) # print([tokenized_input[i] for i in key_indices])
            attentions = raw_attentions[batch_index].mean(dim=1)
            attention_weights = attentions[layer_index, query_index, key_indices]
            num_masks = min(2, attention_weights.shape[0])
            values, indices = torch.topk(attention_weights, k=num_masks, dim=-1)
            indices = indices + offset
            # print(indices)
            print('TOP TOKENS:', [tokenized_input[token_index] for token_index in indices], values.cpu().numpy().tolist())
            # mask tokens with large attention weights
            original_text = original_texts[batch_index]
            print(f'ORIGINAL: {original_text}')
            replacement_token = 'UNIQUEREPLACEMENTTOKEN'
            tokenized_text = tokenized_input.copy() #original_tokenized_input
            replacement_indices = indices# - offset
            for i in replacement_indices:
                if len(set(tokenized_text[i]).intersection(set(string.punctuation)))==0 and (re.sub('|'.join(['▁', ' ']), '', tokenized_text[i]) not in exceptions) and (tokenized_text[i]!='▁'):
                    if tokenized_text[i].startswith('▁'):
                        tokenized_text[i] = ' ' + replacement_token
                    else:
                        tokenized_text[i] = replacement_token
            modified_text = self._tokenizer.convert_tokens_to_string([tokenized_text[i] for i in key_indices])
            print(f'MODIFIED: {modified_text}')
            masked_text, masked_words = token_to_word_mask(original_text, modified_text, replacement_token)
            masked_texts.append(masked_text)
            masked_words_per_text.append(masked_words)
            print(f'MASKED: {masked_text}')
            print(f'MASKED WORDS: {masked_words}')
            print()
      
        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     tokenized_input = tokenized_inputs[batch_index]
        #     query_index = -1
        #     # plot all tokens
        #     key_indices = torch.tensor(range(example_lengths[batch_index])) #print([tokenized_input[i] for i in key_indices])
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-cross-raw-attentions-{batch_index}.png", dpi=500)
        #     # plot original tokens only (without pattern and special tokens)
        #     if tokenized_input[0]=='▁"': 
        #         offset = 1
        #     else:
        #         offset = 0
        #     test_index = -21 - num_pad_tokens[batch_index]
        #     if tokenized_input[test_index]=='"':
        #         end_index = test_index
        #     else:
        #         end_index = test_index + 1
        #     final_index = max([i for i, _ in enumerate(tokenized_input[:end_index])]) + 1
        #     key_indices = torch.tensor(range(offset, final_index)) # print([tokenized_input[i] for i in key_indices])
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-decoder-cross-original-raw-attentions-{batch_index}.png", dpi=500)

        return logits, masked_texts, masked_words_per_text

    def generate(self, input_text: str, **kwargs):
        assert '<extra_id_0>' in input_text
        input_ids = self._tokenizer.encode(input_text, return_tensors='pt').to(self._device)
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(self, input_texts: List[str], toxic_words: List[str], debiasing_prefixes: List[str], 
                                decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False, 
                                min_length: int = None, max_length: int = None, 
                                reg_temperature: float = 1.0, bias_temperature: float = 1.0, already_masked: bool = False, **kwargs) -> List[str]:
        
        num_debiasing_prefixes = len(debiasing_prefixes)
        self._model.init_logits_processor(num_debiasing_prefixes=num_debiasing_prefixes, decay_constant=decay_constant, 
                                          epsilon=epsilon, debug=debug, tokenizer=self._tokenizer, 
                                          reg_temperature=reg_temperature, bias_temperature=bias_temperature)
        
        input_texts = [mask_toxic_words(text, toxic_words, 't5', already_masked=already_masked) for text in input_texts]
        encoder_inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_sentence in input_texts:
                encoder_inputs += [debiasing_prefix + input_sentence]
        
        encoder_inputs = self._tokenizer(encoder_inputs, padding=True, truncation=True, return_tensors='pt').to(self._device)

        output_ids = self._model.generate(inputs=encoder_inputs['input_ids'], 
                                          attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length,
                                          **kwargs)

        batch_size = output_ids.shape[0] // (1 + num_debiasing_prefixes)
        output_ids = output_ids[:batch_size]
        decoded = self._tokenizer.batch_decode(output_ids)
        
        special_tokens = '|'.join(self._tokenizer.all_special_tokens)
        input_texts = [[text] * kwargs['num_return_sequences'] for text in input_texts]
        input_texts = [item for sublist in input_texts for item in sublist]
        for index, seq in enumerate(decoded):
            output_string = seq
            output_match = re.split(r"<extra_id_\d+>", output_string)[1:]
            input_string = input_texts[index]
            # print(f'INPUT: {input_string}')
            # print(f'OUTPUT: {output_string}')
            # print()
            n_sentinel_tokens = len(re.findall(r"<extra_id_(\d+)>", input_string))
            for i, replacement in enumerate(output_match):
                match = re.search(special_tokens, replacement)
                if match:
                    replacement = replacement[:match.span()[0]]
                input_string = re.sub(r"<extra_id_\d+>", replacement, input_string, 1)
                if i == n_sentinel_tokens - 1:
                    break
            decoded[index] = input_string
        
        return decoded

    def generate_self_debiasing_summary(self, input_texts: List[str], toxic_words: List[str], debiasing_prefixes: List[str], 
                                        decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False, 
                                        min_length: int = None, max_length: int = None, 
                                        reg_temperature: float = 1.0, bias_temperature: float = 1.0, **kwargs) -> List[str]:
        
        num_debiasing_prefixes = len(debiasing_prefixes)
        self._model.init_logits_processor(num_debiasing_prefixes=num_debiasing_prefixes, decay_constant=decay_constant, 
                                          epsilon=epsilon, debug=debug, tokenizer=self._tokenizer, 
                                          reg_temperature=reg_temperature, bias_temperature=bias_temperature)
        
        input_texts = [mask_toxic_words(text, toxic_words, 't5') for text in input_texts]
        encoder_inputs = input_texts.copy()     
        for debiasing_prefix in debiasing_prefixes:
            for input_sentence in input_texts:
                encoder_inputs += [debiasing_prefix + input_sentence]
        
        encoder_inputs = ['summarize: ' + text for text in encoder_inputs]
        encoder_inputs = self._tokenizer(encoder_inputs, padding=True, truncation=True, return_tensors='pt').to(self._device)

        n_input_texts = len(input_texts)
        decoder_inputs = [''] * n_input_texts
        for debiasing_prefix in debiasing_prefixes:
            for input_text in [''] * n_input_texts:
                decoder_inputs += [debiasing_prefix + input_text]

        decoder_inputs = self._tokenizer(decoder_inputs, padding=True, truncation=True, return_tensors='pt')
        shifts = decoder_inputs['attention_mask'].shape[-1] - decoder_inputs['attention_mask'].sum(dim=-1) + 1
        for batch_idx in range(decoder_inputs['input_ids'].shape[0]):
            decoder_inputs['input_ids'][batch_idx] = decoder_inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        decoder_inputs['attention_mask'] = torch.where(decoder_inputs['input_ids']==self._model.config.pad_token_id, 0, 1)
        decoder_inputs['input_ids'][:, 0] = 0
        decoder_inputs = {k: v.to(self._device) for k, v in decoder_inputs.items()}
        kwargs['decoder_input_ids'] = decoder_inputs['input_ids']
        kwargs['decoder_attention_mask'] = decoder_inputs['attention_mask']

        input_length = decoder_inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(inputs=encoder_inputs['input_ids'], 
                                          attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length,
                                          **kwargs)

        batch_size = output_ids.shape[0] // (1 + num_debiasing_prefixes)
        output_ids = output_ids[:batch_size]
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded

    def generate_self_debiasing_paraphrase(self, input_texts: List[str], toxic_words: List[str], debiasing_prefixes: List[str], 
                                           decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False, 
                                           min_length: int = None, max_length: int = None, 
                                           reg_temperature: float = 1.0, bias_temperature: float = 1.0, **kwargs) -> List[str]:
        
        num_debiasing_prefixes = len(debiasing_prefixes)
        self._model.init_logits_processor(num_debiasing_prefixes=num_debiasing_prefixes, decay_constant=decay_constant, 
                                          epsilon=epsilon, debug=debug, tokenizer=self._tokenizer, 
                                          reg_temperature=reg_temperature, bias_temperature=bias_temperature)
        
        # input_texts = [mask_toxic_words(text, toxic_words, 't5') for text in input_texts]
        encoder_inputs = input_texts.copy()     
        for debiasing_prefix in debiasing_prefixes:
            for input_sentence in input_texts:
                encoder_inputs += [debiasing_prefix + input_sentence]
        
        encoder_inputs = ['paraphrase: ' + text for text in encoder_inputs]
        encoder_inputs = self._tokenizer(encoder_inputs, padding=True, truncation=True, return_tensors='pt').to(self._device)

        n_input_texts = len(input_texts)
        decoder_inputs = [''] * n_input_texts
        for debiasing_prefix in debiasing_prefixes:
            for input_text in [''] * n_input_texts:
                decoder_inputs += [debiasing_prefix + input_text]

        decoder_inputs = ['paraphrasedoutput: ' + text for text in decoder_inputs]
        decoder_inputs = self._tokenizer(decoder_inputs, padding=True, truncation=True, return_tensors='pt')
        shifts = decoder_inputs['attention_mask'].shape[-1] - decoder_inputs['attention_mask'].sum(dim=-1) + 1
        for batch_idx in range(decoder_inputs['input_ids'].shape[0]):
            decoder_inputs['input_ids'][batch_idx] = decoder_inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        decoder_inputs['attention_mask'] = torch.where(decoder_inputs['input_ids']==self._model.config.pad_token_id, 0, 1)
        decoder_inputs['input_ids'][:, 0] = 0
        decoder_inputs = {k: v.to(self._device) for k, v in decoder_inputs.items()}
        kwargs['decoder_input_ids'] = decoder_inputs['input_ids']
        kwargs['decoder_attention_mask'] = decoder_inputs['attention_mask']

        input_length = decoder_inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(inputs=encoder_inputs['input_ids'], 
                                          attention_mask=encoder_inputs['attention_mask'], 
                                          min_length=min_length, max_length=max_length,
                                          **kwargs)

        batch_size = output_ids.shape[0] // (1 + num_debiasing_prefixes)
        output_ids = output_ids[:batch_size]
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded


class BertWrapper(ModelWrapper):
    """A wrapper for the BERT model"""

    def __init__(self, model_name: str = 'bert-large-uncased', use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained BERT model (default: "bert-base-uncased")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertForMaskedLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        return logits

    def query_model_batch_dev(self, input_texts: List[str], pattern: str):
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        
        # inputs['input_ids'].shape
        # torch.Size([3, 37])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        ### attentions
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
        raw_attentions = torch.stack(outputs.attentions, dim=1)
        # raw_attentions.shape
        # [3, 24, 16, 37, 37] --> [batch_size, num_layers, num_heads, input_length, input_length]
        batch_size, num_layers, num_heads, input_length, _ = raw_attentions.shape
        example_lengths = input_length - (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        for batch_index in range(batch_size):
            print(f'=== INPUT {batch_index} ===')
            mask_token_index = masked_index[1][batch_index]
            tokenized_input = tokenized_inputs[batch_index]
            for layer_index in range(num_layers):
                print(f'=== LAYER {layer_index} ===')
                attention_weights = raw_attentions[batch_index, layer_index, :, mask_token_index, :]
                values, indices = torch.topk(attention_weights, k=5, dim=-1)
                for head_index in range(num_heads): 
                    print([tokenized_input[token_index] for token_index in indices[head_index]], values[head_index].cpu().numpy().tolist())
                print() 
            print()

        # plot raw_attentions
        for batch_index in range(batch_size):
            # plot all tokens
            query_index = masked_index[1][batch_index]
            key_indices = torch.tensor(range(example_lengths[batch_index]))
            fig, ax = plt.subplots()
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-raw-attentions-{batch_index}.png", dpi=500)
            # plot original tokens only (without pattern tokens)
            fig, ax = plt.subplots()
            original_tokenized_input = original_tokenized_inputs[batch_index]
            key_indices = torch.tensor(range(2, len(original_tokenized_input)))
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-original-raw-attentions-{batch_index}.png", dpi=500)


        for batch_index in range(batch_size):
            example_length = example_lengths[batch_index]
            attentions = raw_attentions[batch_index, :, :, :example_length, :example_length] # attentions.shape = [num_layers, num_heads, example_length, example_length]
            # average over num_heads
            attentions = attentions.sum(dim=1) / attentions.shape[1] # attentions.shape = [num_layers, example_length, example_length]
            # add residual weights
            residual_weights = torch.eye(example_length)[None] # residual_weights.shape = [1, example_length, example_length]
            residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
            # CHECK: maybe use softmax for normalization?
            # normalize attention weights
            residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
            residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
            # compute rollout_attentions
            rollout_attentions = compute_rollout_attentions(residual_attentions)
            # plot residual_attentions and rollout_attentions
            query_index = masked_index[1][batch_index]
            original_tokenized_input = original_tokenized_inputs[batch_index]
            all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
            for name, attentions in all_attentions.items():
                # plot all tokens
                fig, ax = plt.subplots()
                key_indices = torch.tensor(range(example_lengths[batch_index]))
                plot_attentions(attentions, key_indices, tokenized_inputs[batch_index], query_index=query_index, head_average=False, ax=ax)
                ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-{name}-{batch_index}.png", dpi=500)
                # plot original tokens only (without pattern tokens)
                fig, ax = plt.subplots()
                key_indices = torch.tensor(range(2, len(original_tokenized_input)))
                plot_attentions(attentions, key_indices, tokenized_inputs[batch_index], query_index=query_index, head_average=False, ax=ax)
                ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-original-{name}-{batch_index}.png", dpi=500)

        return logits

    def generate(self, input_text: str, **kwargs) -> str:
        raise NotImplementedError()

    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        raise NotImplementedError()


class RobertaWrapper(ModelWrapper):
    """A wrapper for the RoBERTa model"""

    def __init__(self, model_name: str = 'roberta-large', use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained RoBERTa model (default: "roberta-large")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self._model = RobertaForMaskedLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        return logits

    def query_model_batch_dev(self, input_texts: List[str], pattern: str):
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        
        # inputs['input_ids'].shape
        # torch.Size([3, 37])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        ### attentions
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
        raw_attentions = torch.stack(outputs.attentions, dim=1)
        # raw_attentions.shape
        # [3, 24, 16, 37, 37] --> [batch_size, num_layers, num_heads, input_length, input_length]
        batch_size, num_layers, num_heads, input_length, _ = raw_attentions.shape
        example_lengths = input_length - (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        # for batch_index in range(batch_size):
        #     print(f'=== INPUT {batch_index} ===')
        #     mask_token_index = masked_index[1][batch_index]
        #     tokenized_input = tokenized_inputs[batch_index]
        #     for layer_index in range(num_layers):
        #         print(f'=== LAYER {layer_index} ===')
        #         attention_weights = raw_attentions[batch_index, layer_index, :, mask_token_index, :]
        #         values, indices = torch.topk(attention_weights, k=5, dim=-1)
        #         for head_index in range(num_heads): 
        #             print([tokenized_input[token_index] for token_index in indices[head_index]], values[head_index].cpu().numpy().tolist())
        #         print() 
        #     print()

        # # plot raw_attentions
        # for batch_index in range(batch_size):
        #     # plot all tokens
        #     query_index = masked_index[1][batch_index]
        #     key_indices = torch.tensor(range(example_lengths[batch_index]))
        #     fig, ax = plt.subplots()
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-raw-attentions-{batch_index}.png", dpi=500)
        #     # plot original tokens only (without pattern tokens)
        #     fig, ax = plt.subplots()
        #     original_tokenized_input = original_tokenized_inputs[batch_index]
        #     key_indices = torch.tensor(range(2, len(original_tokenized_input)))
        #     plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
        #     ax.set_title(original_texts[batch_index])
        #     plt.tight_layout()
        #     fig.savefig(f"figures/{self._model_name}-original-raw-attentions-{batch_index}.png", dpi=500)
        
        masked_texts = []
        masked_words_per_text = []
        for batch_index in range(batch_size):
            example_length = example_lengths[batch_index]
            attentions = raw_attentions[batch_index, :, :, :example_length, :example_length] # attentions.shape = [num_layers, num_heads, example_length, example_length]
            # average over num_heads
            attentions = attentions.sum(dim=1) / attentions.shape[1] # attentions.shape = [num_layers, example_length, example_length]
            # add residual weights
            residual_weights = torch.eye(example_length)[None].to(self._device) # residual_weights.shape = [1, example_length, example_length]
            residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
            # CHECK: maybe use softmax for normalization?
            # normalize attention weights
            residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
            residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
            # compute rollout_attentions
            rollout_attentions = compute_rollout_attentions(residual_attentions)
            # find out which tokens in the original sentence need to be masked
            tokenized_input = tokenized_inputs[batch_index]
            original_tokenized_input = original_tokenized_inputs[batch_index]
            layer_index = -1
            query_index = masked_index[1][batch_index]
            # solves issue with text that starts with special symbols alphanumeric characters
            # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            if tokenized_input[1]=='"': 
                offset = 2
            else:
                offset = 1
            key_indices = torch.tensor(range(offset, offset+len(original_tokenized_input)-2))
            attention_weights = rollout_attentions[layer_index, query_index, key_indices]
            num_masks = min(2, attention_weights.shape[0])
            values, indices = torch.topk(attention_weights, k=num_masks, dim=-1)
            indices = indices + offset
            # print(indices)
            print('TOP TOKENS:', [tokenized_input[token_index] for token_index in indices], values.cpu().numpy().tolist())
            # mask tokens with high rollout attention weights
            original_text = original_texts[batch_index]
            print(f'ORIGINAL: {original_text}')
            replacement_token = 'UNIQUEREPLACEMENTTOKEN'
            tokenized_text = original_tokenized_input[1:-1]
            replacement_indices = indices - offset
            for i in replacement_indices:
                # if not tokenized_text[i].replace('Ġ', '') in string.punctuation:
                if len(set(tokenized_text[i]).intersection(set(string.punctuation)))==0:
                    if tokenized_text[i].startswith('Ġ'):
                        tokenized_text[i] = 'Ġ' + replacement_token
                    else:
                        tokenized_text[i] = replacement_token
            modified_text = self._tokenizer.convert_tokens_to_string(tokenized_text)
            print(f'MODIFIED: {modified_text}')
            masked_text, masked_words = token_to_word_mask(original_text, modified_text, replacement_token)
            masked_texts.append(masked_text)
            masked_words_per_text.append(masked_words)
            print(f'MASKED: {masked_text}')
            print(f'MASKED WORDS: {masked_words}')
            print()
            # # plot residual_attentions and rollout_attentions
            # all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
            # for name, attentions in all_attentions.items():
            #     # plot all tokens
            #     fig, ax = plt.subplots()
            #     key_indices = torch.tensor(range(example_lengths[batch_index]))
            #     plot_attentions(attentions, key_indices, tokenized_input, query_index=query_index, head_average=False, ax=ax)
            #     ax.set_title(original_texts[batch_index])
            #     plt.tight_layout()
            #     fig.savefig(f"figures/{self._model_name}-{name}-{batch_index}.png", dpi=500)
            #     # plot original tokens only (without pattern tokens)
            #     fig, ax = plt.subplots()
            #     key_indices = torch.tensor(range(2, len(original_tokenized_input)))
            #     plot_attentions(attentions, key_indices, tokenized_input, query_index=query_index, head_average=False, ax=ax)
            #     ax.set_title(original_texts[batch_index])
            #     plt.tight_layout()
            #     fig.savefig(f"figures/{self._model_name}-original-{name}-{batch_index}.png", dpi=500)

        return logits, masked_texts, masked_words_per_text

    def generate(self, input_text: str, **kwargs) -> str:
        raise NotImplementedError()

    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        raise NotImplementedError()


class ElectraWrapper(ModelWrapper):
    """A wrapper for the ELECTRA model"""

    def __init__(self, model_name: str = 'google/electra-large-generator', use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained ELECTRA model (default: "google/electra-large-generator")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self._model = ElectraForMaskedLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        return logits

    def query_model_batch_dev(self, input_texts: List[str], pattern: str):
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        masked_index = (inputs['input_ids'] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        logits = outputs.logits[masked_index]
        
        # inputs['input_ids'].shape
        # torch.Size([3, 37])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        ### attentions
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.MaskedLMOutput
        raw_attentions = torch.stack(outputs.attentions, dim=1)
        # raw_attentions.shape
        # [3, 24, 16, 37, 37] --> [batch_size, num_layers, num_heads, input_length, input_length]
        batch_size, num_layers, num_heads, input_length, _ = raw_attentions.shape
        example_lengths = input_length - (inputs['input_ids'] == self._tokenizer.pad_token_id).sum(dim=-1)
        for batch_index in range(batch_size):
            print(f'=== INPUT {batch_index} ===')
            mask_token_index = masked_index[1][batch_index]
            tokenized_input = tokenized_inputs[batch_index]
            for layer_index in range(num_layers):
                print(f'=== LAYER {layer_index} ===')
                attention_weights = raw_attentions[batch_index, layer_index, :, mask_token_index, :]
                values, indices = torch.topk(attention_weights, k=5, dim=-1)
                for head_index in range(num_heads): 
                    print([tokenized_input[token_index] for token_index in indices[head_index]], values[head_index].cpu().numpy().tolist())
                print() 
            print()

        # plot raw_attentions
        for batch_index in range(batch_size):
            # plot all tokens
            query_index = masked_index[1][batch_index]
            key_indices = torch.tensor(range(example_lengths[batch_index]))
            fig, ax = plt.subplots()
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-raw-attentions-{batch_index}.png", dpi=500)
            # plot original tokens only (without pattern tokens)
            fig, ax = plt.subplots()
            original_tokenized_input = original_tokenized_inputs[batch_index]
            key_indices = torch.tensor(range(2, len(original_tokenized_input)))
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_inputs[batch_index], query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-original-raw-attentions-{batch_index}.png", dpi=500)


        for batch_index in range(batch_size):
            example_length = example_lengths[batch_index]
            attentions = raw_attentions[batch_index, :, :, :example_length, :example_length] # attentions.shape = [num_layers, num_heads, example_length, example_length]
            # average over num_heads
            attentions = attentions.sum(dim=1) / attentions.shape[1] # attentions.shape = [num_layers, example_length, example_length]
            # add residual weights
            residual_weights = torch.eye(example_length)[None] # residual_weights.shape = [1, example_length, example_length]
            residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
            # CHECK: maybe use softmax for normalization?
            # normalize attention weights
            residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
            residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
            # compute rollout_attentions
            rollout_attentions = compute_rollout_attentions(residual_attentions)
            # plot residual_attentions and rollout_attentions
            query_index = masked_index[1][batch_index]
            original_tokenized_input = original_tokenized_inputs[batch_index]
            all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
            for name, attentions in all_attentions.items():
                # plot all tokens
                fig, ax = plt.subplots()
                key_indices = torch.tensor(range(example_lengths[batch_index]))
                plot_attentions(attentions, key_indices, tokenized_inputs[batch_index], query_index=query_index, head_average=False, ax=ax)
                ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-{name}-{batch_index}.png", dpi=500)
                # plot original tokens only (without pattern tokens)
                fig, ax = plt.subplots()
                key_indices = torch.tensor(range(2, len(original_tokenized_input)))
                plot_attentions(attentions, key_indices, tokenized_inputs[batch_index], query_index=query_index, head_average=False, ax=ax)
                ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-original-{name}-{batch_index}.png", dpi=500)

        return logits

    def generate(self, input_text: str, **kwargs) -> str:
        raise NotImplementedError()

    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        raise NotImplementedError()


class GPT2Wrapper(ModelWrapper):

    def __init__(self, model_name: str = "gpt2-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._model_name = model_name.replace('/', '-')
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name).to(self._device)
        if self._device == 'cuda':
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id
        self._model.eval()

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1
        output = self._model(**inputs)['logits']
        return torch.stack([output[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])

    def query_model_batch_dev(self, input_texts: List[str], pattern: str, p_masks: float):
        pattern_start_index = -(len(pattern)-len('<INPUT>')-1)
        original_texts = [text[1:pattern_start_index] for text in input_texts]
        inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        logits = outputs.logits
        
        # inputs['input_ids'].shape
        # torch.Size([3, 30])
        input_strings = self._tokenizer.batch_decode(inputs['input_ids'])
        batch_size = inputs['input_ids'].shape[0]
        tokenized_inputs = []
        tokenized_outputs = []
        original_tokenized_inputs = []
        for i in range(batch_size):
            tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            tokenized_outputs.append(self._tokenizer.convert_ids_to_tokens(logits[i].argmax(dim=-1)))
            original_ids = self._tokenizer(original_texts[i]).input_ids
            original_tokenized_inputs.append(self._tokenizer.convert_ids_to_tokens(original_ids))

        logits = torch.stack([logits[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])

        ### attentions
        # https://huggingface.co/docs/transformers/master/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        raw_attentions = torch.stack(outputs.attentions, dim=1)
        # raw_attentions.shape
        # [3, 48, 25, 30, 30] --> [batch_size, num_layers, num_heads, input_length, input_length]
        batch_size, num_layers, num_heads, input_length, _ = raw_attentions.shape
        example_lengths = inputs['attention_mask'].sum(dim=-1)

        # plot raw_attentions
        for batch_index in range(batch_size):
            # plot all tokens
            tokenized_input = tokenized_inputs[batch_index]
            query_index = output_indices[batch_index]
            key_indices = torch.tensor(range(example_lengths[batch_index]))
            fig, ax = plt.subplots()
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
            ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-raw-attentions-{batch_index}.png", dpi=500)
            # plot original tokens only (without pattern tokens)
            fig, ax = plt.subplots()
            original_tokenized_input = original_tokenized_inputs[batch_index]
            # solves issue with text that starts with special symbols which are not alphanumeric characters
            # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            if tokenized_input[0]=='"': 
                offset = 1
            else:
                offset = 0
            key_indices = torch.tensor(range(offset, offset+len(original_tokenized_input)))
            plot_attentions(raw_attentions[batch_index], key_indices, tokenized_input, query_index=query_index, ax=ax)
            # ax.set_title(original_texts[batch_index])
            plt.tight_layout()
            fig.savefig(f"figures/{self._model_name}-original-raw-attentions-{batch_index}.png", dpi=500)
        
        masked_texts = []
        masked_words_per_text = []
        for batch_index in range(batch_size):
            example_length = example_lengths[batch_index]
            attentions = raw_attentions[batch_index, :, :, :example_length, :example_length] # attentions.shape = [num_layers, num_heads, example_length, example_length]
            # average over num_heads
            attentions = attentions.sum(dim=1) / attentions.shape[1] # attentions.shape = [num_layers, example_length, example_length]
            # add residual weights
            residual_weights = torch.eye(example_length)[None].to(self._device) # residual_weights.shape = [1, example_length, example_length]
            residual_attentions = attentions + residual_weights # residual_attentions.shape = [num_layers, example_length, example_length]
            # CHECK: maybe use softmax for normalization?
            # normalize attention weights
            residual_sum = residual_attentions.sum(dim=-1)[..., None] # residual_sum.shape = [num_layers, example_length, 1]
            residual_attentions = residual_attentions / residual_sum # residual_attentions.shape = [num_layers, example_length, example_length]
            # compute rollout_attentions
            rollout_attentions = compute_rollout_attentions(residual_attentions)
            # find out which tokens in the original sentence need to be masked
            tokenized_input = tokenized_inputs[batch_index]
            original_tokenized_input = original_tokenized_inputs[batch_index]
            layer_index = 0
            query_index = output_indices[batch_index]
            # solves issue with text that starts with special symbols which are not alphanumeric characters
            # example: original_text = '''"Kid's testimony is just no damn good 'tall."'''
            if tokenized_input[0]=='"': 
                offset = 1
            else:
                offset = 0
            key_indices = torch.tensor(range(offset, offset+len(original_tokenized_input)))
            attention_weights = attentions[layer_index, query_index, key_indices]
            num_masks = math.ceil(p_masks * attention_weights.shape[0])
            num_masks = min(num_masks, attention_weights.shape[0])
            values, indices = torch.topk(attention_weights, k=num_masks, dim=-1)
            indices = indices + offset
            # print('TOP TOKENS:', [tokenized_input[token_index] for token_index in indices], values.cpu().numpy().tolist())
            # mask tokens with high rollout attention weights
            original_text = original_texts[batch_index]
            # print(f'ORIGINAL: {original_text}')
            replacement_token = 'UNIQUEREPLACEMENTTOKEN'
            tokenized_text = original_tokenized_input
            replacement_indices = indices - offset
            for i in replacement_indices:
                # if not tokenized_text[i].replace('Ġ', '') in string.punctuation:
                if len(set(tokenized_text[i]).intersection(set(string.punctuation)))==0 and re.sub('|'.join(['Ġ', ' ']), '', tokenized_text[i]) not in exceptions:
                    if tokenized_text[i].startswith('Ġ'):
                        tokenized_text[i] = 'Ġ' + replacement_token
                    else:
                        tokenized_text[i] = replacement_token
            modified_text = self._tokenizer.convert_tokens_to_string(tokenized_text)
            # print(f'MODIFIED: {modified_text}')
            masked_text, masked_words = token_to_word_mask(original_text, modified_text, replacement_token)
            masked_texts.append(masked_text)
            masked_words_per_text.append(masked_words)
            # print(f'MASKED: {masked_text}')
            # print(f'MASKED WORDS: {masked_words}')
            # print()
            # plot residual_attentions and rollout_attentions
            all_attentions = {'residual-attentions': residual_attentions, 'rollout-attentions': rollout_attentions}
            for name, attentions in all_attentions.items():
                # plot all tokens
                fig, ax = plt.subplots()
                key_indices = torch.tensor(range(example_lengths[batch_index]))
                plot_attentions(attentions, key_indices, tokenized_input, query_index=query_index, head_average=False, ax=ax)
                ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-{name}-{batch_index}.png", dpi=500)
                # plot original tokens only (without pattern tokens)
                fig, ax = plt.subplots()
                original_tokenized_input = original_tokenized_inputs[batch_index]
                key_indices = torch.tensor(range(1, len(original_tokenized_input)+1))
                plot_attentions(attentions, key_indices, tokenized_input, query_index=query_index, head_average=False, ax=ax)
                # ax.set_title(original_texts[batch_index])
                plt.tight_layout()
                fig.savefig(f"figures/{self._model_name}-original-{name}-{batch_index}.png", dpi=500)

        return logits, masked_texts, masked_words_per_text


    def generate(self, input_text: str, **kwargs):
        raise NotImplementedError()


    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, min_length: int = None, max_length: int = None,
                                **kwargs) -> List[str]:
        raise NotImplementedError()
