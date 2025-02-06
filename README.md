# BLUE
- Code for [``Rethinking Residual Distribution in Locate-then-Edit Model Editing``]

## Requirements
**At least one A40 48G GPU. Better A800 80G GPU.**

- pytorch
- einops==0.4.0
- higher==0.2.1
- hydra-core==1.2.0
- transformers==4.23.1
- datasets==2.12.0
- matplotlib==3.6.1
- spacy==3.4.1
- scipy==1.11.4
- scikit-learn==1.0.2
- nltk==3.7

## Quick Start
### An example for editing Llama3 (8B) on counterfact dataset using MEMIT$_{\text{BLUE}}$
#### 1. Edit Llama3 (8B) model 
 
    python3 -m experiments.evaluate     --alg_name=MEMIT     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-blue.json --ds_name=mcf --dataset_size_limit=2000    --num_edits=100 --downstream_eval_steps=5

This command runs an evaluation script for the MEMIT$_{\text{BLUE}}$ algorithm using the Llama3-8b-instruct. Below are the explanations for each argument:

- `--alg_name=MEMIT`: Specifies the name of the algorithm being used, which is MEMIT in this case.
- `--model_name=meta-llama/Meta-Llama-3-8B-Instruct`: Indicates the name of the model being evaluated, here it is Llama-3-8B-Instruct.
- `--hparams_fname=Llama3-8B-blue.json`: Points to the JSON file containing hyperparameters specific to the Llama-3-8B-Instruct model. If specifying the original MEMIT, use `Llama3-8B.json`. If specifying MEMIT$_{\text{BLUE}}$, use `Llama3-8B-blue.json`. Other models allow this scheme.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=2000`: Sets the total number of editing samples to 2000.
- `--num_edits=100`: Defines the batch size for each round of editing, meaning 100 edits will be performed in each batch. 
- `--downstream_eval_steps=5`: indicates that a test of general capabilities is conducted after every 5 rounds of editing.
#### 2. Summarize the results

    python summarize.py --dir_name=MEMIT --runs=run_<run1>,run_<run2>

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git).
