# Data directories

The original West African Taro Leaf Blight dataset is not committed because it
contains 18,248 images and is approximately 1.27 GB. The notebook downloads the
versioned dataset reproducibly from:

`Project-AgML/taro_blight_stage_classification`

The empty `train/`, `validation/`, and `test/` directories preserve the
assignment's required repository layout. Their exact stratified membership is
stored in `models/data_split_indices.npz`.

The `uploaded/` directory is reserved for labelled images submitted through the
retraining workflow. Runtime uploads are ignored by Git.

The dataset is distributed under CC BY 4.0. The small reproducibility samples
under `load_tests/samples/` retain that license and are attributed to the
dataset authors through the source dataset record.
