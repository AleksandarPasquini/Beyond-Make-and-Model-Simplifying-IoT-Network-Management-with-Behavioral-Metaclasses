import tensorflow as tf
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, LayerNormalization, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding
from transformers import GPT2Tokenizer
import os
import json


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
    

def data_generator(file_path, 
                   batch_size, 
                   max_length, 
                   pad_token_id=0, 
                   limit_per_mac=100000,
                   return_mac=False):
    """
    If return_mac=True, yields ( (input_ids_batch, attention_mask_batch), labels_batch, macs_batch ).
    Else, yields ( (input_ids_batch, attention_mask_batch), labels_batch ).
    """
    mac_counts = {}  # Dictionary to keep track of how many inputs we've seen per MAC
    
    sequences = []
    labels = []
    macs_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            label = data['label'] 
            mac = data['mac']  # this is a list of MAC addresses

            # If we haven't seen a MAC before, initialize its count to 0

            if mac not in mac_counts:
                mac_counts[mac] = 0
                # Skip if we've already reached the limit for this MAC
            if mac_counts[mac] >= limit_per_mac:
                continue
            mac_counts[mac] += 1

            y_true_one_hot = tf.one_hot(label, depth=7)

            # Collect the data
            sequences.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            labels.append(y_true_one_hot.numpy())
            
            # For debugging or test-time, keep track of the MAC addresses
            # We'll just store them as a list-of-lists in `macs_list`.
            macs_list.append(mac)

            # Once we hit batch_size, yield the batch
            if len(sequences) >= batch_size:
                yield_batch = _package_batch(sequences, labels, macs_list, max_length, return_mac)
                
                # Yield depending on return_mac
                if return_mac:
                    yield yield_batch  # ( (input_ids_batch, attention_mask_batch), labels_batch, macs_batch )
                else:
                    yield yield_batch[:2]  # ( (input_ids_batch, attention_mask_batch), labels_batch )

                # Reset for next batch
                sequences = []
                labels = []
                macs_list = []


def _package_batch(sequences, labels, macs_list, max_length, return_mac):
    """
    Helper function that packages sequences and labels (and optionally macs) into tensors/batches.
    """
    input_ids_batch = tf.constant([seq['input_ids'] for seq in sequences], dtype=tf.int32)
    attention_masks_batch = tf.constant([seq['attention_mask'] for seq in sequences], dtype=tf.int32)
    labels_batch = tf.constant(labels, dtype=tf.float32)
    
    features_dict = {
        'input_ids': input_ids_batch,
        'attention_mask': attention_masks_batch
    }
    
    if return_mac:
        # Convert list-of-lists of MACs into a tf.string tensor. 
        # Each entry might have multiple MACs, so we join them with comma or just keep them as is.
        # Here, we will create a single string per sample by joining them:
        macs_batch = tf.constant(macs_list, dtype=tf.string)
        return (features_dict, labels_batch, macs_batch)
    else:
        return (features_dict, labels_batch, None)
                
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

max_length = 1000
classes = [0,1,2,3,4,5,6]
batch_size = 25
files = ["/scratch/apasquini/Data/LSIF/LSIF_1tokens.json",
        "/scratch/apasquini/Data/Aalto/Aalto_1tokens.json",
        "/scratch/apasquini/Data/Deakin/Deakin_1tokens.json",
        "/scratch/apasquini/Data/CIC/CIC_1tokens.json",
        "/scratch/apasquini/Data/UNSW/UNSW_1tokens.json"]
train_file = "/scratch/apasquini/Data/UNSW/UNSW_1tokens.json"
smooths = [0]
for smooth in smooths:
    print("********************************")
    print("Training")
    print(os.path.basename(os.path.dirname(train_file)))
    print(smooth)
    print("********************************")


    # Create the training dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_file, batch_size, max_length),
        output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.float32),
        output_shapes=(
            {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
            (None, 7)
        )
    ).prefetch(tf.data.AUTOTUNE)

    num_labels = 7
    '''

    classes = [0,1,2,3,4,5,6]
    # Filter and limit each class
    balanced_datasets = []
    min_class_count = 10000
    for class_label in classes:
        # Unbatch the dataset
        unbatched_dataset = train_dataset.unbatch()
        
        # Apply the filter to the unbatched dataset
        class_ds = unbatched_dataset.filter(lambda x, y: tf.equal(y, class_label))
        
        # Take 'min_class_count' samples from each class
        class_ds = class_ds.take(min_class_count)
        
        # Do not batch here; collect unbatched datasets
        balanced_datasets.append(class_ds)

    # Concatenate the datasets
    balanced_train_dataset = balanced_datasets[0]
    for ds in balanced_datasets[1:]:
        balanced_train_dataset = balanced_train_dataset.concatenate(ds)

    # Shuffle and batch
    balanced_train_dataset = balanced_train_dataset.shuffle(buffer_size=min_class_count * len(balanced_datasets))
    balanced_train_dataset = balanced_train_dataset.batch(batch_size)
    '''
    # Create a distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # 1. Define multiple input tensors
        input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

        # 2. Pass only 'input_ids' to the embedding layer
        x = TokenAndPositionEmbedding(
            vocabulary_size=50257,     # or your actual vocabulary size 
            sequence_length=max_length,
            embedding_dim=64
        )(input_ids)

        # 3. Apply a stack of TransformerDecoders
        #    If you want to apply the attention mask, pass it as `padding_mask`:
        x = TransformerEncoder(intermediate_dim=128, num_heads=2, dropout=0.2, normalize_first=True)(
            x, padding_mask=attention_mask
        )
        x = TransformerEncoder(intermediate_dim=96, num_heads=2, dropout=0.2, normalize_first=True)(
            x, padding_mask=attention_mask
        )
        x = TransformerEncoder(intermediate_dim=64, num_heads=2, dropout=0.2, normalize_first=True)(
            x, padding_mask=attention_mask
        )

        x = LayerNormalization()(x)
        pooled_max = GlobalMaxPooling1D()(x)
        pooled_avg = GlobalAveragePooling1D()(x)
        x = Concatenate()([pooled_max, pooled_avg])


        outputs = Dense(7, activation="softmax")(x)

        # 4. Build a functional model that accepts both inputs
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

        # 5. Compile the model
        model.compile(
            optimizer='AdamW',
            loss= tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth),
            metrics=['accuracy']
        )

    # Train the model
    history = model.fit(
        train_dataset
    )

    for test_file in files:
        if train_file == test_file:
            continue
        mac_label_counts = defaultdict(lambda: defaultdict(int))
        print("********************************")
        print("Testing")
        print(os.path.basename(os.path.dirname(test_file)))
        print("********************************")

        test_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(test_file, batch_size, max_length, return_mac=False),
            output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.float32),
            output_shapes=(
                {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
                (None, 7)
            )
        ).prefetch(tf.data.AUTOTUNE)

        test_loss, test_accuracy = model.evaluate(test_dataset, batch_size=20)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")