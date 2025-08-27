import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from transformers import GPT2Tokenizer
from collections import defaultdict
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
    


max_length = 1000
classes = [0,1,2,3,4,5,6]
batch_size = 8
files = ["/scratch/apasquini/Data/LSIF/LSIF_1d.npz",
        "/scratch/apasquini/Data/Aalto/Aalto_1d.npz",
        "/scratch/apasquini/Data/Deakin/Deakin_1d.npz",
        "/scratch/apasquini/Data/CIC/CIC_1d.npz",
        "/scratch/apasquini/Data/UNSW/UNSW_1d.npz"]
train_file = "/scratch/apasquini/Data/UNSW/UNSW_1d.npz"
smooths = [0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
mac_5 = False
lab_5 = True
for smooth in smooths:
    print("********************************")
    print("Training")
    print(os.path.basename(os.path.dirname(train_file)))
    print(smooth)
    print("********************************")
    
    data = np.load(train_file)
    
    train=data['inputs']
    labels=data['labels']
    mac=data['macs']

    filtered_train = []
    filtered_labels = []

    mac_counts = defaultdict(int)
    mac_buffers = {}
    lab_buffers = {}
    for inp, label, m in zip(train, labels, mac):
        if mac_counts[m] < 100000:
            if mac_5:
                if m not in mac_buffers:
                    mac_buffers[m] = []
                
                # Add the current packet/label to this MAC's buffer
                mac_buffers[m].append((inp, label))
                
                # If we have 5 packets buffered for this MAC, add them at once
                if len(mac_buffers[m]) == 5:
                    truncated_inps = [b_inp[:200] for (b_inp, _) in mac_buffers[m]]
                    combined_inp = tf.concat(truncated_inps, axis=0)  # shape [1000]
                    filtered_train.append(combined_inp)
                    filtered_labels.append(tf.one_hot(label, depth=7))
                    mac_counts[m] += 5
                    mac_buffers[m] = []
            elif lab_5: 
                if label not in lab_buffers:
                    lab_buffers[label] = []
                
                # Add the current packet/label to this MAC's buffer
                lab_buffers[label].append((inp, m))
                
                # If we have 5 packets buffered for this MAC, add them at once
                if len(lab_buffers[label]) == 5:
                    truncated_inps = [b_inp[:200] for (b_inp, _) in lab_buffers[label]]
                    combined_inp = tf.concat(truncated_inps, axis=0)  # shape [1000]
                    filtered_train.append(combined_inp)
                    filtered_labels.append(tf.one_hot(label, depth=7))
                    for (_, m) in lab_buffers[label]:
                        mac_counts[m] += 1
                    lab_buffers[label] = []

            else:
                filtered_train.append(inp)
                filtered_labels.append(tf.one_hot(label, depth=7))
                mac_counts[m] += 1

    del data

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
    embedding_dim = 64
    filters = 128
    with strategy.scope():
        model = Sequential()
        model.add(Embedding(256, embedding_dim))
        model.add(Conv1D(filters, kernel_size=15))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(embedding_dim, name="first_dense_output"))
        model.add(Dense(7, activation='softmax'))  # Add a dense layer for classification
        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth), metrics=['accuracy'])

    epochs = 1
    filtered_train = np.array(filtered_train, dtype="int32")
    filtered_labels = np.array(filtered_labels, dtype="int32")

    # Train the model
    history = model.fit(
        filtered_train,
        filtered_labels,
        batch_size=20
    )
    # Save the fine-tuned model
    #model.save_pretrained('./1d_tuned_model_tf')
    #print("Fine-tuned model saved to './1d_tuned_model_tf'")


    for test_file in files:
        if train_file == test_file:
            continue

        
        data = np.load(test_file)
            
        test=data['inputs']
        labels=data['labels']
        mac=data['macs']
        filtered_test = []
        filtered_labels = []

        mac_counts = defaultdict(int)
        mac_buffers = {}
        lab_buffers = {}
        for inp, label, m in zip(test, labels, mac):
            if mac_counts[m] < 100000:
                if mac_5:
                    if m not in mac_buffers:
                        mac_buffers[m] = []
                    
                    # Add the current packet/label to this MAC's buffer
                    mac_buffers[m].append((inp, label))
                    
                    # If we have 5 packets buffered for this MAC, add them at once
                    if len(mac_buffers[m]) == 5:
                        truncated_inps = [b_inp[:200] for (b_inp, _) in mac_buffers[m]]
                        combined_inp = tf.concat(truncated_inps, axis=0)  # shape [1000]
                        filtered_test.append(combined_inp)
                        filtered_labels.append(tf.one_hot(label, depth=7))
                        mac_counts[m] += 5
                        mac_buffers[m] = []
                elif lab_5: 
                    if label not in lab_buffers:
                        lab_buffers[label] = []
                    
                    # Add the current packet/label to this MAC's buffer
                    lab_buffers[label].append((inp, m))
                    
                    # If we have 5 packets buffered for this MAC, add them at once
                    if len(lab_buffers[label]) == 5:
                        truncated_inps = [b_inp[:200] for (b_inp, _) in lab_buffers[label]]
                        combined_inp = tf.concat(truncated_inps, axis=0)  # shape [1000]
                        filtered_test.append(combined_inp)
                        filtered_labels.append(tf.one_hot(label, depth=7))
                        for (_, m) in lab_buffers[label]:
                            mac_counts[m] += 1
                        lab_buffers[label] = []
            else:
                filtered_test.append(inp)
                filtered_labels.append(tf.one_hot(label, depth=7))
                mac_counts[m] += 1

        del data
        
        # Access the saved arrays
        filtered_test = np.array(filtered_test, dtype="int32")
        filtered_labels = np.array(filtered_labels, dtype="int32")

        # Evaluate the model on the testing dataset
        test_loss, test_accuracy = model.evaluate(filtered_test, filtered_labels, batch_size=20)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")