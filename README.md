# French-G2P-model

Fine-tuned a French Grapheme-to-Phoneme (G2P) model using ByT5-small on 600K French grapheme-phoneme pairs from the GLAFF dictionary.

ByT5 was chosen for its byte-level tokenization, which handles French accented characters without any preprocessing.

## Approach

1. Fine-tuned ByT5-small on 600K pairs from GLAFF dictionary (learning_rate: 4e-4, 3 epochs)

2. Evaluated using Phoneme Error Rate (PER) against multilingual G2P baseline

## Results
Fine tune model result:

{'eval_loss': '0.01528', 'eval_runtime': '109.2', 'eval_samples_per_second': '493.8', 'eval_steps_per_second': '15.44', 'epoch': '3'}
100% 45510/45510 [3:09:58<00:00,  3.99it/s]
100% 1686/1686 [01:49<00:00, 15.23it/s]
```                                       
==================================================
Model                                         PER
==================================================
french_g2p_model_final                     0.1245 <- 54% improvement over baseline
byt5-multilingual (baseline)               0.2691
byt5-base                                 74.5940
==================================================
```
## Project Structure
```
French-G2P-model
├── README.md
├── main_training_model.py  # Fine-tunes ByT5-small on GLAFF
├── evaluation_model.py
├── french_g2p_eval.csv
└── utils/
    ├── __init__.py
    └── data_processing_glaff.py
```
## Stack
Python, HuggingFace, pandas, transformers and pytorch

## Reference
GLAFF Dictionary : http://redac.univ-tlse2.fr/lexiques/glaff.html
ByT5 paper: https://arxiv.org/abs/2105.13626
