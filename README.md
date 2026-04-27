# NLP Project

This repository contains several NLP experiments for analyzing maintenance work order text. The main project workflow uses the `NLP Report*.csv` files to predict trade categories and work order costs from short text descriptions. The repo also contains exploratory transformer notebooks for BERT and ELECTRA-style language modeling.

## Repository Contents

| File | Purpose |
| --- | --- |
| `nlp_baseline_notebook.ipynb` | Baseline TF-IDF experiments for trade classification and cost prediction. |
| `nlp_word2vec_experiment_notebook.ipynb` | Compares TF-IDF models with averaged Word2Vec embeddings. |
| `nlp_word2vec_experiment_notebook.html` | Rendered HTML version of the Word2Vec experiment notebook. |
| `domain_aware_cost_experiment_finished.ipynb` | Cost prediction experiment using domain-aware text processing and hand-built maintenance features. |
| `BERT-fine-tuning.ipynb` | BERT fine-tuning experiment for a separate text classification dataset. |
| `electra.ipynb` | ELECTRA/DistilBERT-style masked language modeling experiment. |
| `electra.py` | Helper dataset and model sanity-check utilities used by `electra.ipynb`. |
| `data.zip` | Zipped maintenance report CSV files. |
| `requirements.txt` | Base notebook/scikit-learn environment exported from the working environment. |

## Project Goals

The main maintenance work order experiments ask:

1. Can we predict the work order trade/category from the text description?
2. Can we predict work order cost from the same text?
3. Do Word2Vec embeddings improve over TF-IDF?
4. Does adding facilities-domain knowledge improve cost prediction?

The strongest classical workflow is based on TF-IDF features with scikit-learn models:

- Trade classification: `TfidfVectorizer` plus `LogisticRegression` or `LinearSVC`
- Cost prediction: `TfidfVectorizer` plus `Ridge`
- Domain-aware cost prediction: TF-IDF plus binary/numeric features for maintenance language such as replacement, repair, emergency, HVAC, electrical, plumbing, and scope indicators

## Data

The project expects three CSV files:

```text
NLP Report 2002-2009.csv
NLP Report 2010-2019.csv
NLP Report.csv
```

These files are stored inside `data.zip` under a `data/` directory:

```text
data/
  NLP Report 2002-2009.csv
  NLP Report 2010-2019.csv
  NLP Report.csv
```

Most notebooks currently load the CSV files by name from the project root. After extracting `data.zip`, copy the CSV files from `data/` into the project root, or edit the notebook paths to include `data/`.

Expected root layout after setup:

```text
NLP_Project/
  NLP Report 2002-2009.csv
  NLP Report 2010-2019.csv
  NLP Report.csv
  nlp_baseline_notebook.ipynb
  nlp_word2vec_experiment_notebook.ipynb
  domain_aware_cost_experiment_finished.ipynb
  ...
```

## Environment Setup

The project was developed as a notebook-based Python project. Python 3.10 or 3.11 is recommended.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the base requirements

```bash
pip install -r requirements.txt
```

If `pip` fails on the `packaging @ file:///...` line, remove that one line from `requirements.txt` and install `packaging` normally:

```bash
pip install packaging
pip install -r requirements.txt
```

### 3. Install optional experiment dependencies

The base `requirements.txt` covers the classical scikit-learn notebooks, but some experiments need extra packages.

For the Word2Vec notebook:

```bash
pip install gensim
```

For the BERT and ELECTRA notebooks:

```bash
pip install torch transformers tqdm tokenizers torchtest
```

If you have an NVIDIA GPU, install the PyTorch build that matches your CUDA version from the official PyTorch install selector. The CPU build will work, but transformer notebooks will run much more slowly.

### 4. Register the notebook kernel

```bash
python -m ipykernel install --user --name nlp-project --display-name "Python (NLP Project)"
```

### 5. Extract the project data

Windows PowerShell:

```powershell
Expand-Archive -Path data.zip -DestinationPath .
Copy-Item ".\data\NLP Report*.csv" .
```

macOS/Linux:

```bash
unzip data.zip
cp data/NLP\ Report*.csv .
```

## Running the Code

Start JupyterLab from the project root:

```bash
jupyter lab
```

Then select the `Python (NLP Project)` kernel and run the notebooks from top to bottom.

### Recommended run order

1. `nlp_baseline_notebook.ipynb`
2. `nlp_word2vec_experiment_notebook.ipynb`
3. `domain_aware_cost_experiment_finished.ipynb`

This order starts with the simplest reproducible models, then moves to the comparison and domain-aware experiments.

## Example Workflows

### Run the baseline notebook

Use this notebook to reproduce the basic TF-IDF classification and regression experiments.

```bash
jupyter lab nlp_baseline_notebook.ipynb
```

Expected behavior:

- Loads the three `NLP Report*.csv` files
- Cleans and combines work order text fields
- Trains TF-IDF models
- Reports classification metrics such as accuracy and `classification_report`
- Reports regression metrics such as MAE and RMSE

### Run the Word2Vec comparison

```bash
pip install gensim
jupyter lab nlp_word2vec_experiment_notebook.ipynb
```

This notebook compares:

- TF-IDF plus `LinearSVC` for trade classification
- Averaged Word2Vec embeddings plus `LogisticRegression`
- TF-IDF plus `Ridge` for cost prediction
- Averaged Word2Vec embeddings plus `Ridge`

The notebook output shows that TF-IDF performed better than averaged Word2Vec for this dataset, likely because short maintenance descriptions contain high-signal keywords that TF-IDF preserves more directly.

### Run the domain-aware cost experiment

```bash
jupyter lab domain_aware_cost_experiment_finished.ipynb
```

This notebook tests whether facilities-domain knowledge improves cost prediction. It compares:

- Baseline TF-IDF Ridge regression
- Domain-aware TF-IDF plus manually engineered flags and numeric features
- Additional scope and cost-weighted Ridge experiments

The notebook searches for files matching:

```python
glob.glob("NLP Report*.csv")
```

So make sure the CSV files are in the project root before running it.

## Transformer Experiments

The transformer notebooks are less self-contained than the main maintenance-report notebooks.

### `BERT-fine-tuning.ipynb`

This notebook imports `BertTokenizer`, `BertModel`, PyTorch, and `transformers`. It also expects separate files:

```text
X.txt
YL1.txt
```

Those files are not included in `data.zip`. To reproduce this notebook, place `X.txt` and `YL1.txt` in the project root before running the notebook.

### `electra.ipynb`

This notebook uses Hugging Face ELECTRA components and the helper code in `electra.py`. It expects BabyLM-style text files under paths such as:

```text
babylm_data/babylm_10M/
babylm_data/babylm_test/
```

Those BabyLM files are not included in this repository. To reproduce the ELECTRA experiment, download or prepare the BabyLM data and place it in the expected folder structure, then run:

```bash
jupyter lab electra.ipynb
```

The helper file `electra.py` defines:

- `Dataset`: a PyTorch dataset that reads text files and creates masked-language-modeling inputs
- `test_model`: a sanity check that verifies trainable parameters change, frozen parameters stay fixed, and no parameters become NaN or Inf

## Reproducibility Notes

- Run notebooks from the project root so relative paths resolve correctly.
- The main CSV notebooks expect the `NLP Report*.csv` files in the root directory.
- Some notebook outputs are already saved, so rerunning cells may take time and may produce slightly different train/test splits if random seeds are changed.
- Large transformer models may download pretrained weights the first time they run.
- CPU execution is fine for the scikit-learn notebooks. GPU execution is strongly recommended for BERT/ELECTRA.

## Troubleshooting

### `FileNotFoundError: NLP Report 2002-2009.csv`

Extract `data.zip` and copy the CSV files into the project root:

```bash
unzip data.zip
cp data/NLP\ Report*.csv .
```

On Windows PowerShell:

```powershell
Expand-Archive -Path data.zip -DestinationPath .
Copy-Item ".\data\NLP Report*.csv" .
```

### `ModuleNotFoundError: No module named 'gensim'`

Install the Word2Vec dependency:

```bash
pip install gensim
```

### `ModuleNotFoundError: No module named 'transformers'`

Install the transformer dependencies:

```bash
pip install torch transformers tqdm tokenizers torchtest
```

### Jupyter does not show the virtual environment

Register the kernel again:

```bash
python -m ipykernel install --user --name nlp-project --display-name "Python (NLP Project)"
```

### `BERT-fine-tuning.ipynb` cannot find `X.txt` or `YL1.txt`

Those files are not included in this repository. Add them to the project root or skip that notebook if you only want to reproduce the maintenance work order experiments.

### `electra.ipynb` cannot find `babylm_data`

The BabyLM dataset is not included in this repository. Add the expected BabyLM folder structure or skip the ELECTRA notebook.
