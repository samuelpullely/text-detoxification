import argparse
import os
import random
from typing import List, Dict
import re
from collections import Counter

from scipy.stats import pearsonr
from tqdm import tqdm

from modeling import ModelWrapper, BartWrapper, T5Wrapper, BertWrapper, RobertaWrapper, ElectraWrapper, GPT2Wrapper
from io_utils import load_model_outputs, ModelOutput, load_toxic_words

PATTERNS = {
    'toxicity': '"<INPUT>"\nQuestion: Does the above text contain rude, disrespectful, or unreasonable language?\nAnswer:',
    # 'severe_toxicity': '"<INPUT>"\nQuestion: Does the above text contain very hateful, aggressive, disrespectful language?\nAnswer:',
    # 'sexually_explicit': '"<INPUT>"\nQuestion: Does the above text contain sexually explicit language?\nAnswer:',
    # 'threat': '"<INPUT>"\nQuestion: Does the above text contain a threat?\nAnswer:',
    # 'identity_attack': '"<INPUT>"\nQuestion: Does the above text contain negative or hateful language targeting someone because of their identity?\nAnswer:',
    # 'profanity': '"<INPUT>"\nQuestion: Does the above text contain swear words, curse words, or other obscene or profane language?\nAnswer:',
}

MODELS = {
    'bart': BartWrapper,
    't5': T5Wrapper,
    'gpt2': GPT2Wrapper,
    'bert': BertWrapper,
    'roberta': RobertaWrapper,
    'electra': ElectraWrapper,
}

MASK_TOKENS = {
    'bart': '<mask>',
    't5': ' <extra_id_0>',
    'bert': ' [MASK]',
    'electra': ' [MASK]',
    'roberta': '<mask>',
    'gpt2': ''
}

# these thresholds have been determined by running the self diagnosis experiment
TOXICITY_THRESHOLDS = {
    'roberta-large': 0.3,
    'google/t5-v1_1-large': 0.8,
    'google/t5-v1_1-xxl': 0.5,
    'google/t5-v1_1-xl': 0.65,
    'facebook/bart-large': 0.55,
    'gpt2-xl': 0.45
}

def run_self_diagnosis_experiment(wrapper: ModelWrapper, examples: List[str], attribute_name: str, pattern: str, output_dir: str,
                                  output_choices: List[str], toxic_words: List[str], best_threshold: float, batch_size: int = 4, p_masks: float = 0.2) -> Dict[str, float]:
    """
    Runs the self diagnosis experiment from the paper for a single model and attribute.
    :param wrapper: the wrapper for the pretrained language model
    :param examples: the examples to perform self-diagnosis on
    :param attribute_name: the attribute to be considered
    :param pattern: the self-diagnosis pattern to use (this pattern must contain the exact sequence `<INPUT>` exactly once; this sequence
           is replaced with the actual input)
    :param output_choices: the possible output tokens, where the first token must correspond to a positive self-diagnosis (i.e., the given
           input does exhibit the considered attribute)
    :param batch_size: the batch size for processing examples
    :return: a dictionary containing the accuracy and correlation coefficient for the dev and test sets
    """
    predicted_scores = []
    all_masked_texts = []
    all_masked_words_per_text = []
    example_iterator = tqdm(list(chunks(examples, batch_size)), desc="Example batches")

    for example_batch in example_iterator:
        input_texts = [build_input_text(pattern, example) for example in example_batch]
        token_probability_distribution, masked_texts, masked_words_per_text = wrapper.get_token_probability_distribution(
            input_texts, output_choices=output_choices, pattern=pattern, word_masking=True, p_masks=p_masks
        )
        all_masked_texts += masked_texts
        all_masked_words_per_text += masked_words_per_text

        for idx, example in enumerate(example_batch):
            # token_probability_distribution[idx] is of the form [("Yes", p_yes), ("No", p_no)], so we obtain the probability of the input
            # exhibiting the considered attribute by looking at index (0,1)
            predicted_scores.append(token_probability_distribution[idx][0][1])

    toxic_masked_words_per_text = []
    final_masked_texts = []
    for score, words, masked, original in zip(predicted_scores, all_masked_words_per_text, all_masked_texts, examples):
        if score > best_threshold:
            toxic_masked_words_per_text.append(words)
            final_masked_texts.append(masked)
        else:
            final_masked_texts.append(original)
            # get all masked words not only for sentences marked as toxic
            toxic_masked_words_per_text.append(words)


    toxic_percentage = sum([score > best_threshold for score in predicted_scores]) / len(predicted_scores)
    
    toxic_masked_words = [item for sublist in toxic_masked_words_per_text for item in sublist]
    pattern = '|'.join(toxic_words)
    num_toxic_masked_words = len(toxic_masked_words)
    if num_toxic_masked_words==0:
        word_masking_accuracy = 0
    else:
        word_masking_accuracy = sum([(1 if re.search(pattern, word, re.IGNORECASE) else 0) for word in toxic_masked_words]) / num_toxic_masked_words

    unique_toxic_masked_words = sorted(list(set(toxic_masked_words)))
    num_unique_toxic_masked_words = len(unique_toxic_masked_words)

    sentence_detox_accuracy =  1 - sum([(1 if re.search(pattern, text, re.IGNORECASE) else 0) for text in final_masked_texts]) / len(final_masked_texts)

    all_masked_detox_accuracy =  1 - sum([(1 if re.search(pattern, text, re.IGNORECASE) else 0) for text in all_masked_texts]) / len(all_masked_texts)

    baseline_detox_accuracy = 1 - sum([(1 if re.search(pattern, text, re.IGNORECASE) else 0) for text in examples]) / len(examples)

    filename = f'{wrapper._model_name}-masked-{p_masks}'
    output_path = os.path.join(output_dir, f'{filename}.txt')
    with open(output_path, 'w', encoding='utf8') as fh:
        for text in all_masked_texts:
            fh.write(text + '\n')

    # toxic_word_counter = Counter(toxic_masked_words)
    # filename = f'{wrapper._model_name}-masked-toxic-words'
    # output_dir = 'output/masked'
    # output_path = os.path.join(output_dir, f'{filename}.txt')
    # with open(output_path, 'w', encoding='utf8') as fh:
    #     for word, count in toxic_word_counter.most_common():
    #         fh.write(f'{word} {count}\n')

    return {'word_masking_accuracy': word_masking_accuracy, 'all_masked_detox_accuracy': all_masked_detox_accuracy, 'toxic_percentage': toxic_percentage, 
            'sentence_detox_accuracy': sentence_detox_accuracy, 'detox_ratio': sentence_detox_accuracy / toxic_percentage, 
            'baseline_detox_accuracy': baseline_detox_accuracy,
            'num_toxic_masked_words': num_toxic_masked_words, 'num_unique_toxic_masked_words': num_unique_toxic_masked_words}


def get_accuracy(actual_scores: List[float], predicted_scores: List[float], threshold: float):
    """
    Computes the accuracy of a model given actual scores, its predictions, and a classification threshold.
    :param actual_scores: the actual label is considered positive (label=1) if the actual score is above 0.5
    :param predicted_scores: the predicted label is considered positive (label=1) if the predicted score is above the given threshold
    :param threshold: the threshold for computing predicted labels
    :return: the accuracy of the predictions
    """
    assert len(actual_scores) == len(predicted_scores)
    hits = 0
    for actual_score, predicted_score in zip(actual_scores, predicted_scores):
        actual_label = 1 if actual_score > 0.5 else 0
        predicted_label = 1 if predicted_score > threshold else 0
        if actual_label == predicted_label:
            hits += 1
    return hits / len(actual_scores)


def build_input_text(pattern: str, text: str, replace_newlines: bool = True):
    """
    Generates input text for a model from a given self-debiasing pattern and a piece of text.
    :param pattern: the pattern to use (must contain the sequence `<INPUT>` exactly once)
    :param text: the text to insert into the pattern
    :param replace_newlines: whether newlines in the text should be replaced with simple spaces
    :return: the corresponding input text
    """
    assert '<INPUT>' in pattern
    if replace_newlines:
        text = text.replace('\n', ' ')
    return pattern.replace('<INPUT>', text)


def chunks(lst: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to a directory into which results are written")
    parser.add_argument("--output_filename", type=str, required=True,
                        help="Path to a file to which the output of the self-diagnosis experiment is written")
    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--model_type", type=str, default='bart', choices=['bart', 't5', 'bert', 'roberta', 'electra', 'gpt2'],
                        help="The model type to use, must be either 'bart' or 't5'")
    parser.add_argument("--models", type=str, nargs='+', default=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help="The specific models to run self-diagnosis experiments for (e.g., 'gpt2-medium gpt2-large')")
    parser.add_argument("--attributes", nargs='+', default=sorted(PATTERNS.keys()), choices=PATTERNS.keys(),
                        help="The attributes to consider. Supported values are: " + str(PATTERNS.keys()))
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[4],
                        help="The batch sizes to use for each model. This must either be a list of the same size as --models, or a single"
                             "batch size to be used for all models")
    parser.add_argument("--p_masks", type=float, default=0.2)

    args = parser.parse_args()
    print(f"Parameters: {args}")

    if not os.path.exists('figures'):
        os.makedirs('figures')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if isinstance(args.batch_sizes, list):
        assert len(args.batch_sizes) == len(args.models), "There have to be exactly as many batch sizes as models"

    examples = []
    filename = args.input_filename
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            examples.append(line.rstrip())

    paths = ['data/toxic_words/obscene_words.txt', 
            'data/toxic_words/swear_words.txt', 
            'data/toxic_words/bad-words.txt']
    toxic_words = load_toxic_words(paths)
    toxic_words = [word for word in toxic_words if len(word)>2]

    for model_idx, model_name in enumerate(args.models):
        wrapper = MODELS[args.model_type](model_name=model_name)
        batch_size = args.batch_sizes[model_idx] if isinstance(args.batch_sizes, list) else args.batch_sizes

        for attribute in args.attributes:
            pattern = PATTERNS[attribute] + MASK_TOKENS[args.model_type]
            result = run_self_diagnosis_experiment(
                wrapper, examples, attribute_name=attribute, pattern=pattern, output_choices=['Yes', 'No'],
                best_threshold=TOXICITY_THRESHOLDS[model_name], toxic_words=toxic_words, p_masks=args.p_masks,
                output_dir=args.output_dir
            )
            print(f'=== RESULT [{model_name}, {attribute}] ===')
            print(result)
            
            # filename = f"{model_name.replace('/', '-')}-{args.output_filename}-{args.p_masks}.txt"
            # output_path = os.path.join(args.output_dir, filename)
            # with open(output_path, 'w', encoding='utf8') as fh:
            #     fh.write(f'=== RESULT [{model_name}, {attribute}] ===\n')
            #     fh.write(f'{result}\n\n')
