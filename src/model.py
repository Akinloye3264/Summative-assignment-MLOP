"""Load and prepare the classifier checkpoint for fine-tuning."""

from pathlib import Path


def prepare_fine_tuning_model(checkpoint: Path):
    import tensorflow as tf

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training model not found: {checkpoint}")

    model = tf.keras.models.load_model(str(checkpoint), compile=False)
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
