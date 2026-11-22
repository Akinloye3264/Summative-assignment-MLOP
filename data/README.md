# Data directories

The original West African Taro Leaf Blight dataset is not committed because it
contains 18,248 images and is approximately 1.27 GB. The notebook downloads the
versioned dataset reproducibly from:

`Project-AgML/taro_blight_stage_classification`

The `train/` and `test/` folders each contain eight labelled examples,
two per class. Folder names give the labels, and file names include the
source dataset row IDs. Each image belongs to the corresponding notebook
split. The training examples also appear in `load_tests/samples/` for the
upload demo. Keep the test examples out of retraining. The separate
[production evaluation report](../reports/production_evaluation.csv) lists
the 24 held-out rows evaluated on AWS. The full train, validation,
and test split membership is stored in `models/data_split_indices.npz`.

The `uploaded/` directory contains `uploads.sqlite3`, which stores labelled
image bytes for retraining. The app imports uploads from old class folders on
first startup and retains those files as a backup. The app creates `uploaded/`
and `retraining/` when needed. Runtime data is ignored by Git.

Dataset source is the [Project-AgML classification dataset](https://huggingface.co/datasets/Project-AgML/taro_blight_stage_classification).
The included samples retain the original CC BY 4.0 licence and are credited
to [Yinka-Banjo et al.](https://doi.org/10.17632/3knm93dkc5.2).
