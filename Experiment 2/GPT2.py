import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFGPT2ForSequenceClassification
from transformers import GPT2Tokenizer
import os
import json

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
    
def data_generator(file_path, batch_size, max_length, pad_token_id=0, limit_per_mac=100000):
    mac_counts = {}  # Dictionary to keep track of how many inputs we've seen per MAC

    with open(file_path, 'r') as f:
        sequences = []
        labels = []
        for line in f:
            data = json.loads(line)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            label = data['label'] 
            mac = data['mac'][0]

            # If we haven't seen this MAC before, initialize its count to 0
            if mac not in mac_counts:
                mac_counts[mac] = 0

            # Skip if we've already reached the limit for this MAC
            if mac_counts[mac] >= limit_per_mac:
                continue
            
            # Increase the count for this MAC
            mac_counts[mac] += 1
            # Pad or truncate the sequences
            #input_ids = pad_or_truncate(input_ids, max_length, pad_token_id)
            #attention_mask = pad_or_truncate(attention_mask, max_length, pad_token_id)

            sequences.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            labels.append(label)

            if len(sequences) >= batch_size:
                input_ids_batch = tf.constant([seq['input_ids'] for seq in sequences], dtype=tf.int32)
                attention_masks_batch = tf.constant([seq['attention_mask'] for seq in sequences], dtype=tf.int32)
                labels_batch = tf.constant(labels, dtype=tf.int32)
                yield {'input_ids': input_ids_batch, 'attention_mask': attention_masks_batch}, labels_batch
                sequences = []
                labels = []
                
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

max_length = 1000
classes = [0,1,2,3,4,5,6]
batch_size = 10
files = ["/scratch/apasquini/Data/LSIF/LSIF_1_tokens.json",
        "/scratch/apasquini/Data/Aalto/Aalto_1_tokens.json",
        "/scratch/apasquini/Data/Deakin/Deakin_1_tokens.json",
        "/scratch/apasquini/Data/CIC/CIC_1_tokens.json",
        "/scratch/apasquini/Data/UNSW/UNSW_1_tokens.json"]
training = ["/scratch/apasquini/Data/UNSW/UNSW_1_tokens.json"]
for train_file in training:
    print("********************************")
    print("Training")
    print(os.path.basename(os.path.dirname(train_file)))
    print("********************************")

    # Create the training dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_file, batch_size, max_length),
        output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int32),
        output_shapes=(
            {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
            (None,)
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
        model = TFGPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
        model.config.pad_token_id = model.config.eos_token_id
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        # Compile the model within the strategy scope
        model.resize_token_embeddings(len(tokenizer))
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])



    # Train the model
    history = model.fit(
        train_dataset
    )


    for test_file in files:
        if train_file == test_file:
            continue
        print("********************************")
        print("Testing")
        print(os.path.basename(os.path.dirname(test_file)))
        print("********************************")

        test_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(test_file, batch_size, max_length),
            output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int32),
            output_shapes=(
                {'input_ids': (None, max_length), 'attention_mask': (None, max_length)},
                (None,)
            )
        ).prefetch(tf.data.AUTOTUNE)

        # Evaluate the model on the testing dataset
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")