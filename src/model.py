"""
NetRecommender-Capstone
Neural Collaborative Filtering Model (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)

Model Architecture:
✔ User Embedding
✔ Item Embedding
✔ Embedding concatenation
✔ Multi-layer neural network (Config-driven)
✔ Dropout and L2 regularization
✔ Sigmoid output for implicit feedback (0/1)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from utils import load_config


# -----------------------------------------------------------------------------
# Build Neural Collaborative Filtering Model (NCF)
# -----------------------------------------------------------------------------
def build_ncf_model(num_users, num_items, config_path="config/config.yaml"):

    config = load_config(config_path)

    emb_dim = config["model"]["embedding_dim"]
    hidden_units = config["model"]["hidden_units"]
    dropout_rate = config["model"]["dropout_rate"]
    learning_rate = config["model"]["learning_rate"]
    l2_reg = config["model"]["l2_regularization"]

    # ---------------------------
    # Inputs
    # ---------------------------
    user_input = layers.Input(name="user", shape=[], dtype=tf.int32)
    item_input = layers.Input(name="item", shape=[], dtype=tf.int32)

    # ---------------------------
    # Embeddings (Trainable Latent Factors)
    # ---------------------------
    user_embedding = layers.Embedding(
        input_dim=num_users,
        output_dim=emb_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="user_embedding",
    )(user_input)

    item_embedding = layers.Embedding(
        input_dim=num_items,
        output_dim=emb_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="item_embedding",
    )(item_input)

    # Remove extra dimension
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    # ---------------------------
    # Concatenate user & item latent features
    # ---------------------------
    concatenated = layers.Concatenate()([user_vec, item_vec])

    # ---------------------------
    # Deep Neural Network Layers
    # ---------------------------
    x = concatenated
    for units in hidden_units:
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_rate)(x)

    # ---------------------------
    # Output layer (implicit feedback)
    # ---------------------------
    output = layers.Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    model = Model(inputs=[user_input, item_input], outputs=output)

    # Compile
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model
