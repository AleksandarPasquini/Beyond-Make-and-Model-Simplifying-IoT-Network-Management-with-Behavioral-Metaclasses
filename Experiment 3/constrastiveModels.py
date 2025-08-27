# -------------------------------------------------------------
#  Ensemble of Perplexity‑based LM one‑class models +
#  Triplet‑loss (margin = 0.7) contrastive one‑class models
#
#  For each incoming packet (sequence):
#    1.  Evaluate PPL models – retain classes whose perplexity
#        ≤ 70‑th percentile threshold computed on training data.
#    2.  For the retained classes, evaluate the corresponding
#        contrastive classifier (sigmoid score).  Keep classes
#        with prob ≥ 0.5.  If several survive, pick the class
#        with the highest probability.  If none survive, return
#        UNKNOWN (‑1).
#  ------------------------------------------------------------
import os
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
os.environ["KERAS_BACKEND"] = "tensorflow"
import gc, json, random
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from collections import defaultdict
from keras_hub.layers import TransformerDecoder, TokenAndPositionEmbedding
from keras.layers import (Input, Dense, LayerNormalization,
                                     GlobalAveragePooling1D, GlobalMaxPooling1D,
                                     Concatenate)
from keras.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

SAVE_DIR = "/scratch/apasquini/snapshots"   # anywhere on fast local storage
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------
#  Hyper‑parameters & constants
# -------------------------------------------------------------
MAX_LEN              = 999
PPL_THRESHOLD_PCTL   = 70        # %ile for LM perplexity cut‑off
CONTRASTIVE_THRESHOLD = 0.5      # sigmoid probability cut‑off
MARGIN               = 0.3      # triplet‑loss margin
BATCH_SIZE           = 40
VOCAB_SIZE           = 50257
EMBED_DIM            = 64
N_CLASSES            = 7

# -------------------------------------------------------------
#  GPU memory limiting
# -------------------------------------------------------------
CLASS_NAMES = {
    0: "Plug", 1: "Light", 2: "Camera", 3: "Speaker",
    4: "Sensor", 5: "Application", 6: "Hub"
} 

TRAIN_FILE = "/scratch/apasquini/Data/UNSW/UNSW_1tokens.json"
TEST_FILES = [
    "/scratch/apasquini/Data/CIC/CIC_1tokens.json",
    "/scratch/apasquini/Data/LSIF/LSIF_1tokens.json",
    "/scratch/apasquini/Data/Aalto/Aalto_1tokens.json",
    "/scratch/apasquini/Data/Deakin/Deakin_1tokens.json"
]

KEEP_MAP = {            # new class → old index
    0: 2,   # Camera   ← old class 0
    1: 3,   # Speaker  ← old class 2
    2: 5    # Sensor   ← old class 5
}
N_KEEP = len(KEEP_MAP)  # 3

# -------------------------------------------------------------
#  JSON helpers / generators (unchanged except genericised)
# -------------------------------------------------------------

def one_class_data_generator(file_path, batch_size, cl):
    """Teacher‑forcing LM samples for class *cl*"""
    mac_counts = {}
    with open(file_path) as fh:
        X_buf, Y_buf = [], []
        for line in fh:
            d = json.loads(line)
            if d['label'] != cl:
                continue
            m = d['mac']
            mac_counts[m] = mac_counts.get(m, 0)
            if mac_counts[m] >= 100000:
                continue
            mac_counts[m] += 1
            in_ids = d['input_ids'][:-1]
            tgt    = d['input_ids'][1:]
            mask   = d['attention_mask'][:-1]
            X_buf.append({'input_ids': in_ids, 'attention_mask': mask})
            Y_buf.append(tgt)
            if len(X_buf) == batch_size:
                yield ({'input_ids': tf.constant([x['input_ids'] for x in X_buf], tf.int32),
                        'attention_mask': tf.constant([x['attention_mask'] for x in X_buf], tf.int8)},
                       tf.constant(Y_buf, tf.int32))
                X_buf, Y_buf = [], []


def ensemble_test_data_generator(file_path, batch_size):
    """Batches of (inputs, true_label) for inference."""
    mac_counts = {}
    with open(file_path) as fh:
        X_buf, y_buf = [], []
        for line in fh:
            d = json.loads(line)
            m = d['mac']
            mac_counts[m] = mac_counts.get(m, 0)
            if mac_counts[m] >= 100000:
                continue
            mac_counts[m] += 1
            in_ids = d['input_ids'][:-1]       
            in_mask = d['attention_mask'][:-1] 
            X_buf.append({'input_ids': in_ids,
                          'attention_mask': in_mask})
            y_buf.append(d['label'])
            if len(X_buf) == batch_size:
                yield ({'input_ids': tf.constant([x['input_ids'] for x in X_buf], tf.int32),
                        'attention_mask': tf.constant([x['attention_mask'] for x in X_buf], tf.int8)},
                       tf.constant(y_buf, tf.int32))
                X_buf, y_buf = [], []

# -------------------------------------------------------------
#  Model builders – (A) Perplexity LM, (B) Contrastive backbone
# -------------------------------------------------------------

def build_lm(max_len=MAX_LEN, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM):
    ids  = Input((max_len,), dtype=tf.int32, name="input_ids")
    mask = Input((max_len,), dtype=tf.int8, name="attention_mask")
    x = TokenAndPositionEmbedding(vocab_size, max_len, embed_dim)(ids)
    for width in [128, 96, 64]:
        x = TransformerDecoder(intermediate_dim=width, num_heads=2,
                               dropout=0.2, normalize_first=True)(x, decoder_padding_mask=mask)
    x = LayerNormalization()(x)
    logits = Dense(vocab_size)(x)
    return Model([ids, mask], logits)


def build_backbone(max_len=MAX_LEN, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int8, name="attention_mask")

    # 2. Pass only 'input_ids' to the embedding layer
    x = TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,    
        sequence_length=max_len,
        embedding_dim=embed_dim
    )(input_ids)

    # 3. Apply a stack of TransformerDecoders
    #    If you want to apply the attention mask, pass it as `padding_mask`:
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
    pooled_max = GlobalMaxPooling1D()(x)
    pooled_avg = GlobalAveragePooling1D()(x)
    embeddings = Concatenate(name="embeddings")([pooled_max, pooled_avg])
    return Model(inputs=[input_ids, attention_mask], outputs=embeddings)


def build_one_class_classifier(backbone):
    ids  = Input((MAX_LEN,), dtype=tf.int32, name="input_ids")
    mask = Input((MAX_LEN,), dtype=tf.int8, name="attention_mask")
    feats = backbone({'input_ids': ids, 'attention_mask': mask}, training=False)
    out  = Dense(1, activation='sigmoid')(feats)
    return Model([ids, mask], out)

# -------------------------------------------------------------
#  Triplet loss (batch‑all, numerically stable)
# -------------------------------------------------------------

def triplet_loss(labels, embeddings, margin=MARGIN, eps=1e-10):
    dtype = embeddings.dtype               # usually float16/bfloat16 under mixed precision
    margin = tf.cast(margin, dtype)        # 0.7 → same dtype as embeds

    # L2-normalise
    embeddings = embeddings / (tf.norm(embeddings, axis=1, keepdims=True) + eps)

    # Pairwise distances (same dtype as embeddings)
    d = tf.norm(tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0), axis=2)

    # Masks in *the same* dtype
    lbl_eq = tf.cast(tf.equal(tf.expand_dims(labels, 1),
                              tf.expand_dims(labels, 0)), dtype)
    pos_mask = lbl_eq - tf.eye(tf.shape(labels)[0], dtype=dtype)
    neg_mask = 1.0 - lbl_eq

    # Triplet loss
    loss_mat = tf.maximum(tf.expand_dims(d, 2) - tf.expand_dims(d, 1) + margin, 0.)
    loss_mat *= tf.expand_dims(pos_mask, 2) * tf.expand_dims(neg_mask, 1)

    num_pos = tf.reduce_sum(tf.cast(loss_mat > eps, dtype))
    return tf.reduce_sum(loss_mat) / (num_pos + eps)

# -------------------------------------------------------------
#  Training helpers
# -------------------------------------------------------------

def masked_loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 50256), tf.float32)  # mask out padding token
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

@tf.function
def batch_features(inputs):
    """
    Return [batch, 2*N_KEEP] tensor with features for the
    kept classes (Camera, Speaker, Sensor).
    Order of columns   = [ppl_camera, ppl_speaker, ppl_sensor,
                          ctr_camera, ctr_speaker, ctr_sensor]
    """
    ppl = []
    ctr = []
    for new_idx in range(N_KEEP):
        old_idx = KEEP_MAP[new_idx]

        # ------ Perplexity for the kept class --------------------
        logits = perplexity_models[old_idx](inputs, training=False)
        mask = tf.cast(tf.not_equal(inputs['input_ids'][:, 1:], 50256), tf.float32)  
        ce = tf.keras.losses.sparse_categorical_crossentropy(inputs['input_ids'][:, 1:], logits[:, :-1, :], from_logits=True)
        ce_masked = tf.reduce_sum(ce * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        ppl.append(tf.exp(ce_masked))

        # ------ Contrastive probability for the kept class -------
        ctr.append( contrastive_models[old_idx](inputs, training=False)[:, 0] )

    ppl = tf.stack(ppl, axis=1)                     # [batch, 3]
    ctr = tf.stack(ctr, axis=1)                     # [batch, 3]
    return tf.concat([ppl, ctr], axis=1)            # [batch, 6]


def train_lm_for_class(cl):
    """
    Train a language model for class *cl* and return
    (model, 70-th-percentile perplexity threshold).
    """
    ds = tf.data.Dataset.from_generator(
        lambda: one_class_data_generator(TRAIN_FILE, BATCH_SIZE, cl),
        output_types = ({'input_ids': tf.int32, 'attention_mask': tf.int8}, tf.int32),
        output_shapes= ({'input_ids': (None, MAX_LEN), 'attention_mask': (None, MAX_LEN)},
                        (None, MAX_LEN))
    )
    model = build_lm()
    model.compile(
        optimizer="adam",
        loss=masked_loss,
    )
    model.fit(ds, epochs=1)

    # ---------- 70-th-percentile perplexity – micro-batch = 1 ----------
    perps = []
    for (x, y) in ds.unbatch().batch(1).take(10000):   # <-- only 1 sequence at a time
        logits = model(x, training=False)
        mask = tf.cast(tf.not_equal(y, 50256), tf.float32)  
        ce = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        ce_masked = tf.reduce_sum(ce * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        perps.append(tf.exp(ce_masked))
    thres = np.percentile(tf.stack(perps).numpy(), PPL_THRESHOLD_PCTL)
    return model, thres


def balanced_file_batch_generator(file_path, batch_size, cl, limit_per_mac=100000, filter_label=None):
    """
    Reads data from a file and yields balanced batches ensuring that each class is represented 
    with at least two samples. It also applies a per-mac limit to avoid over-representation.
    
    Args:
        file_path (str): Path to the file containing one JSON per line.
        batch_size (int): Desired batch size.
        limit_per_mac (int): Maximum allowed occurrences per mac.
        filter_label (optional): If provided, only samples with data['label'] == filter_label will be used.
        
    Yields:
        A tuple: ({'input_ids': tf.Tensor, 'attention_mask': tf.Tensor}, tf.Tensor) where:
            - 'input_ids' and 'attention_mask' are built from the inputs (excluding the last token)
            - The label tensor is built from the targets (the input shifted by one).
    """
    # First, we load all eligible samples from the file into a dictionary keyed by label.
    label_to_buffer = {}
    mac_counts = {}  # keep track of how many times each mac has been used
    num_classes = 2
    unique_labels = [0, 1]

    samples_per_class = max(2, batch_size // num_classes)
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Optionally filter out samples that do not have the desired label.
            if filter_label is not None and data['label'] != filter_label:
                continue
            m = data['mac']
            mac_counts[m] = mac_counts.get(m, 0)
            if mac_counts[m] < 100000:
                mac_counts[m] += 1
            else:
                continue
            
            # Process sample: use teacher forcing where the input is the sequence minus the last token 
            # and the label is the sequence shifted by one.
            input_ids = data['input_ids']
            if len(input_ids) != 1000:
                continue
            attention_mask = data['attention_mask']
            x_ids = input_ids[:-1]
            x_mask = attention_mask[:-1]
            sample_label = 1 if data['label'] == cl else 0
            
            # Group the sample by its label.
            sample = ({'input_ids': x_ids, 'attention_mask': x_mask}, sample_label)
            if sample_label not in label_to_buffer:
                label_to_buffer[sample_label] = []
            label_to_buffer[sample_label].append(sample)
    
            # List of unique labels and number of classes.
            if (len(list(label_to_buffer.keys())) > 1) and all(len(buffer) >= samples_per_class for buffer in label_to_buffer.values()):
                batch_samples = []
                for lab in unique_labels:
                    buf = label_to_buffer[lab]
                    # If the buffer has more than required, sample without replacement and remove those items.
                    if len(buf) > samples_per_class:
                        # Set a fixed seed for reproducibility
                        random.seed(15)
                        chosen = random.sample(buf, samples_per_class)
                        for sample_item in chosen:
                            buf.remove(sample_item)
                    else:
                        # If the number exactly matches, take all and clear the buffer.
                        chosen = list(buf)
                        label_to_buffer[lab] = []
                    batch_samples.extend(chosen)
                
                # Split features and labels.
                batch_input_ids = [features['input_ids'] for (features, _) in batch_samples]
                batch_attention_masks = [features['attention_mask'] for (features, _) in batch_samples]
                batch_labels = [lab for (_, lab) in batch_samples]
                
                # Convert lists into TensorFlow tensors.
                input_ids_batch = tf.constant(batch_input_ids, dtype=tf.int32)
                attention_masks_batch = tf.constant(batch_attention_masks, dtype=tf.int8)
                labels_batch = tf.constant(batch_labels, dtype=tf.int32)
                
                yield ({'input_ids': input_ids_batch, 'attention_mask': attention_masks_batch},
                        labels_batch)

@tf.function
def train_step(batch_x, batch_y, backbone, optimizer):
    with tf.GradientTape() as tape:
        embeddings = backbone(batch_x, training=True)
        loss = triplet_loss(batch_y, embeddings)
    gradients = tape.gradient(loss, backbone.trainable_variables)
    # Only update if there are valid gradients.
    if not all(g is None for g in gradients):
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables))
    return loss

def train_contrastive_clf(cl):
    ds = tf.data.Dataset.from_generator(
        lambda: balanced_file_batch_generator(TRAIN_FILE, 40, cl),
        output_types=(
            {'input_ids': tf.int32, 'attention_mask': tf.int8},
            tf.int32
        ),
        output_shapes=(
            {'input_ids': (None, MAX_LEN), 'attention_mask': (None, MAX_LEN)},
            (None,)
        )
    ).prefetch(tf.data.AUTOTUNE)

    finite_dataset = ds.take(1000)
    dummy = {
        'input_ids': tf.zeros((1, MAX_LEN), tf.int32),
        'attention_mask': tf.zeros((1, MAX_LEN), tf.int8)
    }
    backbone = build_backbone()
    _ = backbone(dummy, training=True)                 # weights exist
    opt      = keras.optimizers.Adam()
    opt.build(backbone.trainable_variables)
    # Triplet pre‑training (2 epochs)
    # Contrastive training phase.
    for step, (batch_x, batch_y) in enumerate(finite_dataset):
        loss = train_step(batch_x, batch_y, backbone, opt)
    # classifier fine‑tune
    clf = build_one_class_classifier(backbone)
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    clf.fit(finite_dataset, epochs=1)
    return clf

# -------------------------------------------------------------
#  1)  Contrastive one-class classifiers
# -------------------------------------------------------------
contrastive_models = {}

for c in range(N_CLASSES):
    w_path = f"{SAVE_DIR}/cls_con_{c}.weights.h5"

    if os.path.isfile(w_path):                    # ← already trained?
        print(f"→ loading contrastive model for class {c}")
        backbone = build_backbone()               # same topology
        clf_gpu  = build_one_class_classifier(backbone)
        clf_gpu.load_weights(w_path)
    else:
        print(f"→ training contrastive model for class {c}")
        clf_gpu = train_contrastive_clf(c)
        clf_gpu.save_weights(w_path)

    contrastive_models[c] = clf_gpu               # keep in RAM


# -------------------------------------------------------------
#  2)  Perplexity language-models  +  thresholds
# -------------------------------------------------------------
perplexity_models, ppl_thresholds = {}, {}
thres_file = os.path.join(SAVE_DIR, f"{PPL_THRESHOLD_PCTL}_ppl.json")
saved_thres = json.load(open(thres_file)) if os.path.isfile(thres_file) else {}

for c in range(N_CLASSES):
    w_path = f"{SAVE_DIR}/perplex_{c}.weights.h5"

    if os.path.isfile(w_path):                    # ← already trained?
        print(f"→ loading LM for class {c}")
        m = build_lm()
        m.load_weights(w_path)
        t = saved_thres.get(str(c))               # may be None if first run
        if t is None:                             # threshold missing → recompute once
            _, t = train_lm_for_class(c)
            saved_thres[str(c)] = float(t)
            json.dump(saved_thres, open(thres_file, "w"))
        else:
            print(t)
    else:
        print(f"→ training LM for class {c}")
        m, t = train_lm_for_class(c)
        m.save_weights(w_path)
        saved_thres[str(c)] = float(t)
        json.dump(saved_thres, open(thres_file, "w"))

    perplexity_models[c] = m
    ppl_thresholds[c]    = t
    print(f"Class {CLASS_NAMES[c]}  PPL threshold: {t:.4f}")


def remap_labels(y_old):
    """
    Map original 0…6 labels → {0,1,2} (Camera/Speaker/Sensor) or –1 (unknown).
    Vectorised NumPy helper used inside the loops.
    """
    new = np.full_like(y_old, -1)       # start with "I don't know"
    for new_idx, old_idx in KEEP_MAP.items():
        new[y_old == old_idx] = new_idx
    return new

# -------------------------------------------------------------
#  Evaluation loop on each test file
# -------------------------------------------------------------

ds_train = tf.data.Dataset.from_generator(
              lambda: ensemble_test_data_generator(TRAIN_FILE, BATCH_SIZE),
              output_types=({'input_ids': tf.int32,
                             'attention_mask': tf.int8}, tf.int32)
           ).prefetch(tf.data.AUTOTUNE)

X_tr, y_tr = [], []
for inputs, y_true_tf in ds_train:
    X_tr.append(batch_features(inputs).numpy())
    y_tr.append(remap_labels(y_true_tf.numpy()))

X_tr = np.vstack(X_tr)
y_tr = np.concatenate(y_tr)
mask = y_tr >= 0
X_tr, y_tr = X_tr[mask], y_tr[mask]
cw = class_weight.compute_class_weight(class_weight='balanced',
                                       classes=np.unique(y_tr),
                                       y=y_tr)
class_weights = dict(enumerate(cw))

# -------------------------------------------------------------
#  MLP definition  (very small, L2 regularised)
# -------------------------------------------------------------
def build_mlp(input_dim, n_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Normalization(),
        keras.layers.Dense(64,  activation='relu',
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dense(32,  activation='relu',
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.layers[0].adapt(X_tr)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
print(np.unique(y_tr))
mlp = build_mlp(input_dim=2*N_KEEP, n_classes=N_KEEP)
mlp.fit(X_tr, y_tr, batch_size=512, epochs=10, class_weight=class_weights, validation_split=0.1, verbose=2)


print("   …training finished.\n")

for test in TEST_FILES:
    print(f">>> Evaluating on {os.path.basename(test)}")

    total, correct = 0, 0
    full, right = 0, 0
    tp = np.zeros(N_CLASSES, dtype=np.int64)
    fp = np.zeros(N_CLASSES, dtype=np.int64)
    fn = np.zeros(N_CLASSES, dtype=np.int64)
    tn = np.zeros(N_CLASSES, dtype=np.int64)

    ds = tf.data.Dataset.from_generator(
            lambda: ensemble_test_data_generator(test, BATCH_SIZE),
            output_types=({'input_ids': tf.int32,
                           'attention_mask': tf.int8}, tf.int32)
         ).prefetch(tf.data.AUTOTUNE)

    for inputs, y_true_tf in ds:
        X_te   = batch_features(inputs).numpy()
        probs  = mlp.predict(X_te, batch_size=1024, verbose=0)

        y_pred = probs.argmax(axis=1)               # 0/1/2
        y_true = remap_labels(y_true_tf.numpy())    # -1 or 0/1/2

        # assign "I don't know" to predictions with prob < 0.5
        low_conf = probs.max(axis=1) < 0.8
        y_pred[low_conf] = -1

        # ---------- overall accuracy on known classes ----------
        mask_known = y_true >= 0
        correct += np.sum((y_pred == y_true) & mask_known)
        total   += np.sum(mask_known)
        right += np.sum(y_pred == y_true)
        full   += len(y_true)

        # ---------- per-class confusion counts (three classes) --
        for c in range(N_KEEP):
            pos      = (y_true == c)
            neg      = (y_true != c) & mask_known
            pred_pos = (y_pred == c)
            tp[c] += np.sum(pred_pos & pos)
            fp[c] += np.sum(pred_pos & neg)
            fn[c] += np.sum(~pred_pos & pos)
            tn[c] += np.sum(~pred_pos & neg)

    acc = correct / total if total else 0.0
    print(f"File accuracy: {acc:.4f}")
    acc =  right / full if full else 0.0
    print(f"Full accuracy: {acc:.4f}")

    print("Per-class BA:")
    for c in range(N_CLASSES):
        tpr = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        tnr = tn[c] / (tn[c] + fp[c]) if (tn[c] + fp[c]) else 0.0
        ba  = 0.5 * (tpr + tnr)
        print(f"  Class {c}: BA = {ba:.4f}")

print("\nDone.")
