import argparse
import os
import random
import torch
from tqdm import tqdm

from modeling import BartWrapper, T5Wrapper
from io_utils import load_toxic_words

DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    "The following text contains very hateful, aggressive, disrespectful language:\n",
    "The following text contains sexually explicit language:\n",
    "The following text contains a threat:\n",
    "The following text contains negative or hateful language targeting someone because of their identity:\n",
    "The following text contains swear words, curse words, or other obscene or profane language:\n"
]

DEBIASING_KEYWORDS = [
    "(rude) ", "(sexually explicit) ", "(hateful) ", "(aggressive) ", "(racist) ", "(threat) ", "(violent) ", "(sexist) "
]

MODELS = {
    'bart': BartWrapper,
    't5': T5Wrapper
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to a directory into which generated outputs are written")
    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--already_masked", action='store_true')
    parser.add_argument("--model_type", type=str, default='bart', choices=['bart', 't5'],
                        help="The model type to use, must be either 'bart' or 't5'")
    parser.add_argument("--model_name", type=str, default='facebook/bart-large',
                        help="The specific model to run self-debiasing experiments for")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    parser.add_argument("--reg_temperature", type=float, default=1)
    parser.add_argument("--bias_temperature", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to perform sampling instead of greedy decoding")
    parser.add_argument("--min_length", type=int, default=0,
                        help="Minimum length for the generated text")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length for the generated text")
    parser.add_argument("--top_k", type=int,
                        help="Only for sampling. If set, only the top_k tokens with the highest probabilities are considered.")
    parser.add_argument("--top_p", type=float,
                        help="Only for sampling. If set, only the tokens with the highest probabilities that sum up to top_p are considered.")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="The number of beams for beam search")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="The number of sequences to return for each prompt")
    parser.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    parser.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for initializing the random number generator used for sampling")
    
    args = parser.parse_args()
    print(f"Parameters: {args}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    wrapper = MODELS[args.model_type](model_name=args.model_name)
    
    mode = 'debiased'
    debiasing_prefixes = (DEBIASING_PREFIXES if not args.use_keywords else DEBIASING_KEYWORDS) if mode == 'debiased' else []

    texts = []
    filename = args.input_filename
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            texts.append(line.rstrip())
    
    texts = [texts[i:i+args.batch_size] for i in range(0, len(texts), args.batch_size)]

    paths = ['data/toxic_words/obscene_words.txt', 
            'data/toxic_words/swear_words.txt', 
            'data/toxic_words/bad-words.txt']
    toxic_words = load_toxic_words(paths)
    toxic_words = [word for word in toxic_words if len(word)>2]

    output_texts = []
    texts_iterator = tqdm(texts)
    generate_func = ('generate_self_debiasing_summary' if args.model_name.startswith('t5')
                     else 'generate_self_debiasing')
    # generate_func = 'generate_self_debiasing_paraphrase'
    generate_func = getattr(wrapper, generate_func)
    for text in texts_iterator:
        output_texts += generate_func(
            text, toxic_words, debiasing_prefixes=debiasing_prefixes, decay_constant=args.decay_constant, 
            epsilon=args.epsilon, debug=args.debug, min_length=args.min_length, max_length=args.max_length, 
            reg_temperature=args.reg_temperature, bias_temperature=args.bias_temperature,
            do_sample=args.do_sample, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences, 
            top_k=args.top_k, top_p=args.top_p, already_masked=args.already_masked
        )

    filename = (f'decay-constant-{args.decay_constant}-max-length-{args.max_length}-sample-{args.do_sample}'
                f'-top-k-{args.top_k}-top-p-{args.top_p}.txt' if args.do_sample
                else f'decay-constant-{args.decay_constant}-max-length-{args.max_length}.txt')
    if args.already_masked:
        input_filename = os.path.basename(args.input_filename)
        filename = f"{args.model_name.replace('/', '-')}-{filename[:-4]}-{input_filename}"
    else:
        filename = f"{args.model_name.replace('/', '-')}-{filename}"
    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, 'w', encoding='utf8') as fh:
        for text in output_texts:
            fh.write(text + '\n')
