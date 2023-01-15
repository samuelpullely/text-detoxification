# Text Detoxification using Pre-Trained Language Models and Plug-and-Play Generation Methods

This repository contains some of the code used in the master thesis [Text Detoxification using Pre-Trained Language Models and Plug-and-Play Generation Methods](Master-Thesis.pdf). 
The code in this repository is based on the following repository: https://github.com/timoschick/self-debiasing.

## Setup
`transformers` version 4.16.2 and Python 3.8.5

## Self-Debiasing
Run self-debiasing with 

```
python self_debiasing.py \
  --model_type bart \
  --model_name facebook/bart-large \ 
  --decay_constant 50 \
  --input_filename data/toxic_sentences.txt \
  --output_dir output/debiased
```

## Masking Toxic Words using Attentions Weights
In order to mask the toxic words appearing in the sentences in [toxic_sentences.txt](data/toxic_sentences.txt) using attention weights, run 

```
python self_diagnosis_attention.py \
  --model_type gpt2 \
  --models gpt2-xl \
  --p_masks 0.3 \
  --input_filename data/toxic_sentences.txt \
  --output_dir output/masked \
  --output_filename results.txt
```

Read [Text Detoxification using Pre-Trained Language Models and Plug-and-Play Generation Methods](Master-Thesis.pdf) for more details.
