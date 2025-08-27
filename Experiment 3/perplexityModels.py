import os
import json
import sys
import numpy as np 
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, LayerNormalization, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Map class integers to names.
names = {
    0: "Plug",
    1: "Light",
    2: "Camera",
    3: "Speaker",
    4: "Sensor",
    5: "Application",
    6: "Hub"
}
plt.rcParams['pdf.fonttype'] = 42

def one_class_data_generator(file_path, batch_size, cl, max_length=1000, limit_per_mac=100000):
    """
    Yields (inputs, labels) for language model training for a single class (cl).
    """
    mac_counts = {}
    with open(file_path, 'r') as f:
        X_buffer = []
        Y_buffer = []
        for line in f:
            data = json.loads(line)
            if data['label'] != cl:
                continue
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            mac = data['mac']
            mac_counts[mac] = mac_counts.get(mac, 0)
            if mac_counts[mac] < 100000:
                mac_counts[mac] += 1
            else:
                continue
            x_ids = input_ids[:-1]
            y_ids = input_ids[1:]
            x_mask = attention_mask[:-1]
            X_buffer.append({'input_ids': x_ids, 'attention_mask': x_mask})
            Y_buffer.append(y_ids)
            if len(X_buffer) == batch_size:
                input_ids_batch = tf.constant([ex['input_ids'] for ex in X_buffer], dtype=tf.int32)
                attention_masks_batch = tf.constant([ex['attention_mask'] for ex in X_buffer], dtype=tf.int32)
                labels_batch = tf.constant(Y_buffer, dtype=tf.int32)
                yield ({'input_ids': input_ids_batch, 'attention_mask': attention_masks_batch}, labels_batch)
                X_buffer, Y_buffer = [], []

def test_data_generator(file_path, cl, batch_size, max_length=1000, limit_per_mac=100000):
    """
    Yields (inputs, labels, binary_class) for one-class testing.
    For each sample, binary_class is 1 if its label equals cl, else 0.
    """
    mac_counts = {}
    with open(file_path, 'r') as f:
        X_buffer = []
        Y_buffer = []
        labs = []
        macs = []
        for line in f:
            data = json.loads(line)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            mac = data['mac']
            label = data['label']
            binary_label = 1 if label == cl else 0
            mac_counts[mac] = mac_counts.get(mac, 0)
            if mac_counts[mac] < 100000:
                mac_counts[mac] += 1
            else:
                continue
            x_ids = input_ids[:-1]
            y_ids = input_ids[1:]
            x_mask = attention_mask[:-1]
            X_buffer.append({'input_ids': x_ids, 'attention_mask': x_mask})
            Y_buffer.append(y_ids)
            labs.append(binary_label)
            macs.append(mac)
            if len(X_buffer) == batch_size:
                input_ids_batch = tf.constant([ex['input_ids'] for ex in X_buffer], dtype=tf.int32)
                attention_masks_batch = tf.constant([ex['attention_mask'] for ex in X_buffer], dtype=tf.int32)
                labels_batch = tf.constant(Y_buffer, dtype=tf.int32)
                is_class = tf.constant(labs, dtype=tf.int32)
                devices = tf.constant(macs, dtype=tf.string)
                yield ({'input_ids': input_ids_batch, 'attention_mask': attention_masks_batch},
                       labels_batch,
                       is_class,
                       devices)
                X_buffer, Y_buffer, labs, macs = [], [], [], []

def ensemble_test_data_generator(file_path, batch_size, max_length=1000, limit_per_mac=100000):
    """
    Yields (inputs, labels, true_class) for ensemble testing.
    Unlike test_data_generator, we keep the actual class label.
    """
    mac_counts = {}
    with open(file_path, 'r') as f:
        X_buffer = []
        Y_buffer = []
        labels_buffer = []
        macs = []
        for line in f:
            data = json.loads(line)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            true_label = data['label']
            mac = data['mac']
            mac_counts[mac] = mac_counts.get(mac, 0)
            if mac_counts[mac] < 100000:
                mac_counts[mac] += 1
            else:
                continue
            x_ids = input_ids[:-1]
            y_ids = input_ids[1:]
            x_mask = attention_mask[:-1]
            X_buffer.append({'input_ids': x_ids, 'attention_mask': x_mask})
            Y_buffer.append(y_ids)
            labels_buffer.append(true_label)
            macs.append(mac)
            if len(X_buffer) == batch_size:
                input_ids_batch = tf.constant([ex['input_ids'] for ex in X_buffer], dtype=tf.int32)
                attention_masks_batch = tf.constant([ex['attention_mask'] for ex in X_buffer], dtype=tf.int32)
                labels_batch = tf.constant(Y_buffer, dtype=tf.int32)
                true_labels = tf.constant(labels_buffer, dtype=tf.int32)
                devices = tf.constant(macs, dtype=tf.string)
                yield ({'input_ids': input_ids_batch, 'attention_mask': attention_masks_batch},
                       labels_batch,
                       true_labels,
                       devices)
                X_buffer, Y_buffer, labels_buffer, macs = [], [], [], []

@tf.function
def evaluate_model_once_training(model, dataset):
    perplexities_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    mean_ce_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    i = tf.constant(0)
    for batch in dataset:
        inputs, labels = batch   # only 2-tuple
        logits = model(inputs, training=False)
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        perplexities = tf.exp(ce)
        mean_ce_ta = mean_ce_ta.write(i, tf.reduce_mean(ce, axis=0))
        perplexities_all = perplexities_all.write(i, perplexities)
        i += 1

    all_perplexities = perplexities_all.concat()
    mean_ce_per_token = tf.exp(tf.reduce_mean(mean_ce_ta.stack(), axis=0))
    return all_perplexities, mean_ce_per_token


@tf.function
def evaluate_model_once_inference(model, dataset):
    """
    For non-training data: batch is (inputs, labels, true_binary, macs).
    """
    perplexities_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    true_all = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    mac_all = tf.TensorArray(tf.string, size=0, dynamic_size=True)

    i = tf.constant(0)
    for batch in dataset:
        inputs, labels, true_binary, macs = batch
        logits = model(inputs, training=False)
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        perplexities = tf.exp(ce)

        # Save perplexities, labels, MACs
        perplexities_all = perplexities_all.write(i, perplexities)
        true_all = true_all.write(i, true_binary)
        mac_all = mac_all.write(i, macs)

        i += 1

    return (
        perplexities_all.concat(),
        true_all.concat(),
        mac_all.concat()
    )

@tf.function
def compute_ensemble_perplexities(models_list, ensemble_ds):
    # one TensorArray for the per‐step stacks,
    # plus the usual true and mac arrays
    ta_all   = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    true_ta  = tf.TensorArray(tf.int32,   size=0, dynamic_size=True)
    mac_ta   = tf.TensorArray(tf.string,  size=0, dynamic_size=True)

    i = tf.constant(0)
    for inputs, labels, true_labels, macs in ensemble_ds:
        # compute each model’s perplexities for this batch
        perps = []
        for mdl in models_list:
            logits = mdl(inputs, training=False)
            ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            perps.append(tf.exp(ce))

        # shape [num_models, batch_size]
        perps_stack = tf.stack(perps, axis=0)

        ta_all  = ta_all.write(i, perps_stack)
        true_ta = true_ta.write(i, true_labels)
        mac_ta  = mac_ta.write(i, macs)
        i += 1

    # now stack into a single Tensor of shape [steps, num_models, batch_size]
    stacked = ta_all.stack()
    # transpose to [num_models, steps, batch_size]
    trans   = tf.transpose(stacked, perm=[1, 0, 2])
    # flatten the last two dims into [num_models, N]
    N       = tf.shape(trans)[1] * tf.shape(trans)[2]
    per_model_perps = tf.reshape(trans, (len(models_list), N))

    return per_model_perps, true_ta.concat(), mac_ta.concat()



@tf.function
def ensemble_predict_for_threshold(perp_matrix,         # shape [num_models, N]
                                   thresholds,          # shape [num_models]
                                   large_const=1e6):
    """
    perp_matrix : tf.float32 [num_models, N]
    thresholds  : tf.float32 [num_models]
    Returns     : tf.int32   [N]  (model index, or -1 if all > threshold)
    """
    # Broadcast thresholds to [num_models, N]
    thresh_mat = tf.expand_dims(thresholds, axis=1)     # [num_models, 1]

    # Mask out values above threshold with `large_const`
    adjusted = tf.where(perp_matrix <= thresh_mat, perp_matrix,
                        tf.fill(tf.shape(perp_matrix), large_const))

    # For every sample (axis‑0 after transpose) pick the model with min perplexity
    min_indices = tf.argmin(adjusted, axis=0, output_type=tf.int32)      # [N]

    # Identify samples where *all* models were masked
    all_large = tf.reduce_all(tf.equal(adjusted, large_const), axis=0)   # [N]

    # Replace indices by −1 for those samples
    return tf.where(all_large,
                    tf.fill(tf.shape(min_indices), -1),
                    min_indices)



def find_json_files(directory):
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_sequence.json"):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def pad_or_truncate(sequence, max_length, pad_token_id=0):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [pad_token_id] * (max_length - len(sequence))

max_length = 999
batch_size = 40

files = [
    "/scratch/apasquini/Data/LSIF/LSIF_1tokens.json",
    "/scratch/apasquini/Data/Aalto/Aalto_1tokens.json",
    "/scratch/apasquini/Data/Deakin/Deakin_1tokens.json",
    "/scratch/apasquini/Data/CIC/CIC_1tokens.json"
]

train_file = "/scratch/apasquini/Data/UNSW/UNSW_1tokens.json"
classes = [0, 1, 2, 3, 4, 5, 6]
possible_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

# Dictionaries to store the trained one-class models and their thresholds.
ensemble_models = {}
ensemble_thresholds = defaultdict(dict)


for cl in [2]:
    print("********************************")
    print("Training on class:", names[cl])
    print("********************************")
    
    # Create a dataset for training for the given class.
    train_dataset = tf.data.Dataset.from_generator(
        lambda: one_class_data_generator(train_file, batch_size, cl, max_length),
        output_types=(
            {'input_ids': tf.int32, 'attention_mask': tf.int32},
            tf.int32
        ),
        output_shapes=(
            {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
            (None, max_length)
        )
    )

    # Build the one-class model.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Build the one-class model.
        input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
        x = TokenAndPositionEmbedding(
            vocabulary_size=50257,
            sequence_length=max_length,
            embedding_dim=64
        )(input_ids)
        x = TransformerDecoder(intermediate_dim=128, num_heads=2, dropout=0.2, normalize_first=True)(
            x, decoder_padding_mask=attention_mask
        )
        x = TransformerDecoder(intermediate_dim=96, num_heads=2, dropout=0.2, normalize_first=True)(
            x, decoder_padding_mask=attention_mask
        )
        x = TransformerDecoder(intermediate_dim=64, num_heads=2, dropout=0.2, normalize_first=True)(
            x, decoder_padding_mask=attention_mask
        )
        x = LayerNormalization()(x)
        outputs = Dense(50257)(x)
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        model.compile(optimizer='adamw', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    
    # Train the model.
    history = model.fit(train_dataset)    
    print("********************************")
    print("Computing perplexity thresholds on training set for:", names[cl])
    print("********************************")
    # Create a separate dataset for evaluation on the same file.
    train_dataset = tf.data.Dataset.from_generator(
        lambda: one_class_data_generator(train_file, batch_size, cl, max_length),
        output_types=(
            {'input_ids': tf.int32, 'attention_mask': tf.int32},
            tf.int32
        ),
        output_shapes=(
            {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
            (None, max_length)
        )
    )
    train_perplexities, average_token = evaluate_model_once_training(model, train_dataset)
    averages_np = average_token.numpy()
    # Create positions for the x-axis (0 to sequence_length - 1)
    positions = np.arange(averages_np.shape[0])
    # Plot a bar chart where x is the position and y is the average value
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.bar(positions, averages_np)
    plt.yscale('log')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Perplexity (Log-scale)')
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')

    # --- minor ticks + grid ---
    ax.minorticks_on()                                   # enable minor ticks
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':',  alpha=0.3)

    # --- save & clean ---
    fig.savefig(f"{names[cl]}_p_log_distribution.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig) 
    plt.clf()
    train_perplexities = train_perplexities.numpy()
    for thres in possible_thresholds:
        threshold = np.percentile(train_perplexities, thres)
        print(f"{thres}% perplexity threshold for class '{names[cl]}' = {threshold:.4f}")
        ensemble_thresholds[cl][thres] = threshold
    ensemble_models[cl] = model

for test_file in files:
    print("============================================")
    print("Individual model classification evaluation on file:", os.path.basename(test_file))
    print("============================================")
    
    for cl in [2]:
        print("Evaluating individual model for class:", names[cl])
        test_ds = tf.data.Dataset.from_generator(
            lambda: test_data_generator(test_file, cl, batch_size, max_length),
            output_types=(
                {'input_ids': tf.int32, 'attention_mask': tf.int32},
                tf.int32,
                tf.int32,
                tf.string
            ),
            output_shapes=(
                {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
                (None, max_length),
                (None,),
                (None,)
            )
        )
        # Compute the model's perplexities once for the entire test set.
        all_perps, all_true, all_macs = evaluate_model_once_inference(ensemble_models[cl], test_ds)
        all_true = all_true.numpy()
        all_macs = all_macs.numpy()
        all_perps = all_perps.numpy()
        for thres in possible_thresholds:
            preds = np.less_equal(all_perps, ensemble_thresholds[cl][thres])
            overall_bal_acc = balanced_accuracy_score(all_true, preds)
            tn, fp, fn, tp = confusion_matrix(all_true, preds, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            print("TPR:", sensitivity)
            print("TNR:", specificity)
            print(f"Accuracy for class '{names[cl]}' at threshold level '{thres}' model: {overall_bal_acc:.4f}")
            target_names = [f'Not {names[cl]}', f'{names[cl]}']
            print(classification_report(all_true, preds, target_names=target_names, labels=[0, 1]))
            print("ROC AUC:", roc_auc_score(all_true, preds))
            num_pred_not = np.sum(preds == 0)
            num_pred_yes = np.sum(preds == 1)
            print(f"Predicted Not {names[cl]}: {num_pred_not} Packets, {names[cl]}: {num_pred_yes} Packets")

for test_file in files:

    print("============================================")
    print("Test file:", os.path.basename(test_file))
    print("============================================")
    print("============================================")
    print("Ensemble classification on file:", os.path.basename(test_file))
    print("============================================")
    
    # Build an ensemble dataset.
    test_ds = tf.data.Dataset.from_generator(
        lambda: ensemble_test_data_generator(test_file, batch_size, max_length),
        output_types=(
            {'input_ids': tf.int32, 'attention_mask': tf.int32},
            tf.int32, tf.int32, tf.string
        ),
        output_shapes=(
            {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
            (None, max_length), (None,), (None)
        )
    )
    
    # Prepare list of models (ordered by class).
    models_list = [ensemble_models[cl] for cl in sorted(ensemble_models.keys())]
    # REUSE COMPUTATION: Precompute perplexities for each model.
    ensemble_perplexities, ensemble_true, _ = compute_ensemble_perplexities(models_list, test_ds)
    ensemble_true = ensemble_true.numpy()
    for thres in possible_thresholds:
        # Prepare the thresholds for each model as a list (keep the ordering the same).
        threshold_values = [
            tf.constant(ensemble_thresholds[cl][thres], dtype=tf.float32)
            for cl in sorted(ensemble_thresholds.keys())
        ]
        # Compute ensemble predictions using the precomputed perplexities.
        preds = ensemble_predict_for_threshold(ensemble_perplexities, threshold_values, large_const=1e6)
        ensemble_accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, ensemble_true), tf.float32)) * 100.0
        tf.print("At threshold level:", thres)
        tf.print("Ensemble Classification Accuracy:", ensemble_accuracy)