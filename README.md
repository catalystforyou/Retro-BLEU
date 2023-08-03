# Retro-BLEU

This project is aimed to evaluate generated retrosynthesis routes via the Retro-BLEU score.

## Requirement
### Submodules
- aizynthfinder
- PaRoutes
- aizynthtrain(optional if you want to extract negative bigrams)

Please use `git clone --recursive https://github.com/catalystforyou/Retro-BLEU.git` to correctly load the submodules.

### Enviornments

Please follow the instructions in the corresponding submodules when preparing the data.

```
conda create -n Retro-BLEU python=3.8
conda activate Retro-BLEU
pip install reaction-utils
conda install rdkit -c conda-forge
pip install rxnmapper

```

## Data Preparation

### Step 1: Preparaing the patent routes
```
conda activate paroutes-env (see submodule PaRoutes)
mkdir data
cp PaRoutes/data/download_data.py data/
cd data
python download_data.py
cp uspto_template_library.csv ../PaRoutes/data
cd ../PaRoutes/setup
python extract_uspto_data.py --template_library ../data/uspto_template_library.csv --all_reactions ../data/selected_reactions_all.csv
python extract_routes.py --max-workers 24
python analyze_routes.py
cd ../../preprocess
python convert_route.py

```

### Step 2: Extracting reaction templates for patent routes

```
cd preprocess
conda activate Retro-BLEU
python extract_templates.py

(if you want to examine template on other radii for Retro*-190, run the commands below)

sh extract_training_template.sh
python merge_train_templates.py
```

### Step 3: Performing single-step predictions

```
cd preprocess
conda activate aizynth-dev (see submodule aizynthfinder)
python single_step.py
```

### Step 4: Generating retrosynthesis routes for set-n1 & n5 via aizynthfinder

You may use the script `retrosynthesis.py` in `preprocess/regen_routes/` and you can modify the config file and targets to predict.
(There is a generated version under ../routes/{n5routes, n1routes}/{mcts_v2, retrostar_v2} and you don't actually need to regenerate the routes.)


### Step 5: Evaluating bigram overlap

```
cd preprocess
python extract_first_routes.py
```
Then you may use the `evaluate_algorithms.ipynb` to examine different retrosynthesis searching algorithms across PaRoutes & Retro*-190.

### Step 5: Selecting the patent-extracted route from set-n1 & set-n5

```
python generate_vocab_bigrams.py
python calc_acc.py
```