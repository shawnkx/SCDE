# SCDE: Sentence Cloze Dataset with High Quality Distractors From Examinations

Code for the paper:

[Sentence Cloze Dataset with High Quality Distractors From Examinations](https://arxiv.org/abs/2004.12934). Xiang Kong*, Varun Gangal*, and Eduard Hovy. ACL2020.

## Leaderboard
If you have new results, it would be great if you could submit it [here](https://paperswithcode.com/sota/question-answering-on-scde)(https://paperswithcode.com/sota/question-answering-on-scde).
## Dependencies
* Python 3.6+
* Pytorch 1.2
* [Transformers](https://github.com/huggingface/transformers) 2.1.1

## Datasets
* SCDE:
    Please submit a data request [here](https://vgtomahawk.github.io/sced.html). The data will be automatically sent to you. Please also check your spam folder.

## Usage
### Installing the Transformers from the source
    cd transformers
    pip install .
### Preprocessing (get the AP+AN context features)
    python extract_features.py --output_dir all_prev_next_test --input_dir scde_data/ --feature_type apn
### Finetune a BERT-based model in folder transformers
    bash train.sh feature_dir


## Acknowledgement
* The code is adapted from Transformers (https://github.com/huggingface/transformers). Thanks!

## License
MIT



