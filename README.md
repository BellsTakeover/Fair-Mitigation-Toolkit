# Multi-Level Bias Mitigation Toolkit (Random Forest)

This place has a set of tools that you can use to reduce bias in tables of data. These tools work with something called Random Forest models. The goal of these tools is to help make the data, in the tables more fair. You can use these tools to apply techniques to reduce bias in the tables of data using Random Forest models.

The main idea of this toolkit is to let users try out methods that make models fair when they are working with any set of data. This way users do not have to put in rules for each disease when they are using the toolkit. The toolkit is meant to work with any dataset. It does this without needing special instructions for specific diseases. The fairness-aware modeling strategies are what make this toolkit so useful, for users who are working with kinds of data.

------------------------------------------------------------

SUPPORTED TECHNIQUES

------------------------------------------------------------

PRE-PROCESSING

- Relabeling (confident label correction)

- SMOTE (class imbalance correction)

- Latent Variables (PCA-based feature compression)

IN-PROCESSING

- L2 Regularization (Logistic Regression comparison)

- Exponentiated Gradient with Equalized Odds (Fairlearn)

Primary modeling approach: Random Forest

------------------------------------------------------------

PROJECT STRUCTURE

------------------------------------------------------------

src/fair_mitigator/

preprocessing.py

models.py

fairness.py

metrics.py

mitigations/

relabel.py

latent.py

pipelines/

baseline.py

relabel.py

smote.py

latent.py

inprocess.py

configs/

baseline.yaml

relabel.yaml

smote.yaml

latent.yaml

inprocess.yaml

reports/

run_YYYYMMDD_HHMMSS/

------------------------------------------------------------

INSTALLATION

------------------------------------------------------------

Create a virtual environment:

python -m venv .venv

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Or install in editable mode:

pip install -e .

------------------------------------------------------------

RUNNING A PIPELINE

------------------------------------------------------------

Example:

./.venv/bin/python -m fair_mitigator.cli \

The user is talking about pipeline and SMOTE. What they are saying is that they have a pipeline and they are using SMOTE. The pipeline is something that they are working with and SMOTE is a technique they are using. Pipeline and SMOTE are important, for the user.

--data your_dataset.csv \

--config configs/smote.yaml

Available pipelines:

- baseline

- relabel

- smote

- latent

- inprocess

------------------------------------------------------------

DATA REQUIREMENTS

------------------------------------------------------------

- Input must be a CSV file

- You have to pick one column to be the target when you are working with YAML. This column is really important because it is what everything else is based on. When you are setting things up in YAML you need to make sure that one column is defined as the target.

- The other columns, in the list are seen as features of the data unless we say otherwise about columns.

Minimal YAML example:

data:

target_col: outcome

feature_cols: null

categorical_cols: null

drop_cols: []

------------------------------------------------------------

OUTPUT

------------------------------------------------------------

Each time you do a run it makes a folder with a timestamp, on it inside.

reports/

The files that are made may include:

* log files

* data files

* image files

The files that are generated may also have types of files. The files that are made can be very different.

- metrics.json

- model.joblib

If you have fairness turned on you will get a file called fairness_summary.json. This file is only available when fairness is enabled. The fairness_summary.json file is what you will see when fairness is turned on.

- You will get a file called relabeling_replacement_log.csv if you have relabeling turned on.

------------------------------------------------------------

REPRODUCIBILITY

------------------------------------------------------------

We use YAML configuration files to control all the experiments. This way we can keep track of what's happening in all the experiments by using these YAML configuration files. All the experiments are set up with the help of these YAML configuration files.

This includes:

- Model parameters

- SMOTE parameters

- Relabeling thresholds

- Things that are personal to us, like attributes are private information that we do not want to share with everyone. Sensitive attributes, such as our address or phone number are not something we give out easily. We consider attributes to be very important and we try to keep them safe. Sensitive attributes are a part of who we are and we need to protect them.

- Cross-validation settings

No dataset-specific logic is hardcoded.

------------------------------------------------------------

NOTES

------------------------------------------------------------

- They only use SMOTE on the training set, which's the SMOTE technique. The SMOTE method is applied to the training set.

- When you do relabeling it changes the training labels and that is all it does it only modifies the training labels.

- Latent variables are fit on training data only.

- Fairness metrics are evaluated on the untouched test set and changed test set.

------------------------------------------------------------

AUTHOR

------------------------------------------------------------
Isabella Mixton-Garcia

Multi-Level Bias Mitigation Reproducible Toolkit
