import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from transformers import GPT2Tokenizer
from collections import defaultdict
import os
import json




UNSW_Exact = {
    "70:ee:50:18:34:43":0,
    "18:b4:30:25:be:e4":1,
    "70:ee:50:03:b8:ac":2,
    "00:24:e4:1b:6f:96":3,
    "e0:76:d0:33:bb:85":4

}
UNSW_Close = {
    "70:ee:50:18:34:43":0,
    "18:b4:30:25:be:e4":1,
    "70:ee:50:03:b8:ac":2,
    "00:24:e4:1b:6f:96":3,
    "e0:76:d0:33:bb:85":4,
    "44:65:0d:56:cc:d3":5,
    "f4:f2:6d:93:51:f1":6,
    "00:16:6c:ab:6b:88":7,
    "50:c7:bf:00:56:39":8,
    "ec:1a:59:83:28:11":9,
    "74:6a:89:00:2e:25":10,
    "70:5a:0f:e4:9b:c0":11
}
UNSW_Full ={
    "d0:52:a8:00:67:5e":12, # Smart Things
    "44:65:0d:56:cc:d3":5, # Amazon Echo
    "70:ee:50:18:34:43":0, # Netatmo Welcome
    "f4:f2:6d:93:51:f1":6, # TP-Link Day Night Cloud camera
    "00:16:6c:ab:6b:88":7, # Samsung SmartCam
    "30:8c:fb:2f:e4:b2":13, # Dropcam
    "00:62:6e:51:27:2e":14, # Insteon Camera
    "e8:ab:fa:19:de:4f":14, # Insteon Camera
    "00:24:e4:11:18:a8":15, # Withings Smart Baby Monitor
    "ec:1a:59:79:f4:89":16, # Belkin Wemo switch
    "50:c7:bf:00:56:39":8, # TP-Link Smart plug
    "74:c6:3b:29:d7:1d":17, # iHome
    "ec:1a:59:83:28:11":9,  # Belkin wemo motion sensor
    "18:b4:30:25:be:e4":1,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac":2,  # Netatmo weather station
    "00:24:e4:1b:6f:96":3, # Withings Smart scale
    "74:6a:89:00:2e:25":10, # Blipcare Blood Pressure meter
    "00:24:e4:20:28:c6":18, # Withings Aura smart sleep sensor
    "d0:73:d5:01:83:08":19,  # Light Bulbs LiFX Smart Bulb
    "18:b7:9e:02:20:44":20,  # Triby Speaker
    "e0:76:d0:33:bb:85":4, # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0":11, # HP Printer
    "30:8c:fb:b6:ea:45":21  # Nest Dropcam

}

UNSW_Exact_meta = {
    "70:ee:50:18:34:43":0, # Netatmo Welcome
    "18:b4:30:25:be:e4":1,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac":1,  # Netatmo weather station
    "00:24:e4:1b:6f:96":1, # Withings Smart scale
    "e0:76:d0:33:bb:85":2, # PIX-STAR Photo-frame

}
UNSW_Close_meta = {
    "44:65:0d:56:cc:d3":3, # Amazon Echo
    "70:ee:50:18:34:43":0, # Netatmo Welcome
    "f4:f2:6d:93:51:f1":0, # TP-Link Day Night Cloud camera
    "00:16:6c:ab:6b:88":0, # Samsung SmartCam
    "50:c7:bf:00:56:39":4, # TP-Link Smart plug
    "ec:1a:59:83:28:11":1,  # Belkin wemo motion sensor
    "18:b4:30:25:be:e4":1,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac":1,  # Netatmo weather station
    "00:24:e4:1b:6f:96":1, # Withings Smart scale
    "74:6a:89:00:2e:25":1, # Blipcare Blood Pressure meter
    "e0:76:d0:33:bb:85":2, # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0":2, # HP Printer

}
UNSW_Full_meta ={
    "d0:52:a8:00:67:5e":5, # Smart Things
    "44:65:0d:56:cc:d3":3, # Amazon Echo
    "70:ee:50:18:34:43":0, # Netatmo Welcome
    "f4:f2:6d:93:51:f1":0, # TP-Link Day Night Cloud camera
    "00:16:6c:ab:6b:88":0, # Samsung SmartCam
    "30:8c:fb:2f:e4:b2":0, # Dropcam
    "00:62:6e:51:27:2e":0, # Insteon Camera
    "e8:ab:fa:19:de:4f":0, # Insteon Camera
    "00:24:e4:11:18:a8":0, # Withings Smart Baby Monitor
    "ec:1a:59:79:f4:89":4, # Belkin Wemo switch
    "50:c7:bf:00:56:39":4, # TP-Link Smart plug
    "74:c6:3b:29:d7:1d":4, # iHome
    "ec:1a:59:83:28:11":1,  # Belkin wemo motion sensor
    "18:b4:30:25:be:e4":1,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac":1,  # Netatmo weather station
    "00:24:e4:1b:6f:96":1, # Withings Smart scale
    "74:6a:89:00:2e:25":1, # Blipcare Blood Pressure meter
    "00:24:e4:20:28:c6":1, # Withings Aura smart sleep sensor
    "d0:73:d5:01:83:08":6,  # Light Bulbs LiFX Smart Bulb
    "18:b7:9e:02:20:44":3,  # Triby Speaker
    "e0:76:d0:33:bb:85":2, # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0":2, # HP Printer
    "30:8c:fb:b6:ea:45":0  # Nest Dropcam

}


Deakin_Exact = {
    "70:ee:50:57:95:29":0,
    "cc:a7:c1:6a:b5:78":1,
    "70:ee:50:96:bb:dc":2,
    "00:24:e4:e4:55:26":3,
    "00:24:e4:e3:15:6e":3,
    "b0:02:47:6f:63:37":4
}
Deakin_Close = {
    "70:ee:50:57:95:29":0,
    "cc:a7:c1:6a:b5:78":1,
    "70:ee:50:96:bb:dc":2,
    "00:24:e4:e4:55:26":3,
    "00:24:e4:e3:15:6e":3,
    "b0:02:47:6f:63:37":4,
    "40:f6:bc:bc:89:7b":5,
    "54:af:97:bb:8d:8f":6,
    "00:16:6c:d7:d5:f9":7,
    "10:5a:17:b8:a2:0b":8,
    "10:5a:17:b8:9f:70":8,
    "fc:67:1f:53:fa:6e":9, # Perfk Motion Sensor
    "1c:90:ff:bf:89:46":9,
    "00:24:e4:f6:91:38":10,# Withings Connect (Blood Pressure)
    "00:24:e4:f7:ee:ac":10,
    "84:69:93:27:ad:35":11
}
Deakin_Full ={
    "40:f6:bc:bc:89:7b":5,# Echo Dot (4th Gen)
    "68:3a:48:0d:d4:1c":12,# Aeotec Smart Hub
    "70:ee:50:57:95:29":0,# Netatmo Smart Indoor Security Camera
    "54:af:97:bb:8d:8f":6,# TP-Link Tapo Pan/Tilt Wi-Fi Camera
    "70:09:71:9d:ad:10":13, #32' Smart Monitor M80B UHD
    "00:16:6c:d7:d5:f9":7, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
    "40:ac:bf:29:04:d4":14, # EZVIZ Security Camera
    "10:5a:17:b8:a2:0b":8, #TOPERSUN Smart Plug
    "10:5a:17:b8:9f:70":8, #TOPERSUN Smart Plug
    "fc:67:1f:53:fa:6e":9, # Perfk Motion Sensor
    "1c:90:ff:bf:89:46":9,# Perfk Motion Sensor
    "cc:a7:c1:6a:b5:78":1,# NEST Protect smoke alarm
    "70:ee:50:96:bb:dc":2,# Netatmo Weather Station
    "00:24:e4:e3:15:6e":3,# Withings Body+ (Scales)
    "00:24:e4:e4:55:26":3,# Withings Body+ (Scales)
    "00:24:e4:f6:91:38":10,# Withings Connect (Blood Pressure)
    "00:24:e4:f7:ee:ac":10,# Withings Connect (Blood Pressure)
    "70:3a:2d:4a:48:e2":15, # TUYA Smartdoor Bell
    "b0:02:47:6f:63:37":4,# Pix-Star Easy Digital Photo Frame
    "84:69:93:27:ad:35":11,# HP Envy
    "18:48:be:31:4b:49":16,# Echo Show 8
    "74:d4:23:32:a2:d7":16,# Echo Show 8
    "6e:fe:2f:5a:d7:7e":17,# GALAXY Watch5 Pro
    "90:48:6c:08:da:8a":18, # Ring Video Doorbell

}

Deakin_Exact_meta = {
    "70:ee:50:57:95:29":0,
    "cc:a7:c1:6a:b5:78":1,
    "70:ee:50:96:bb:dc":1,
    "00:24:e4:e4:55:26":1,
    "00:24:e4:e3:15:6e":1,
    "b0:02:47:6f:63:37":2

}
Deakin_Close_meta = {
    "70:ee:50:57:95:29":0,
    "cc:a7:c1:6a:b5:78":1,
    "70:ee:50:96:bb:dc":1,
    "00:24:e4:e4:55:26":1,
    "00:24:e4:e3:15:6e":1,
    "b0:02:47:6f:63:37":2,
    "40:f6:bc:bc:89:7b":3,
    "54:af:97:bb:8d:8f":0,
    "00:16:6c:d7:d5:f9":0,
    "10:5a:17:b8:a2:0b":4,
    "10:5a:17:b8:9f:70":4,
    "fc:67:1f:53:fa:6e":1, # Perfk Motion Sensor
    "1c:90:ff:bf:89:46":1,
    "00:24:e4:f6:91:38":1,# Withings Connect (Blood Pressure)
    "00:24:e4:f7:ee:ac":1,
    "84:69:93:27:ad:35":2

}
Deakin_Full_meta = {
"40:f6:bc:bc:89:7b":3,# Echo Dot (4th Gen)
"68:3a:48:0d:d4:1c":5,# Aeotec Smart Hub
"70:ee:50:57:95:29":0,# Netatmo Smart Indoor Security Camera
"54:af:97:bb:8d:8f":0,# TP-Link Tapo Pan/Tilt Wi-Fi Camera
"70:09:71:9d:ad:10":2, #32' Smart Monitor M80B UHD
"00:16:6c:d7:d5:f9":0, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
"40:ac:bf:29:04:d4":0, # EZVIZ Security Camera
"10:5a:17:b8:a2:0b":4, #TOPERSUN Smart Plug
"10:5a:17:b8:9f:70":4, #TOPERSUN Smart Plug
"fc:67:1f:53:fa:6e":1, # Perfk Motion Sensor
"1c:90:ff:bf:89:46":1,# Perfk Motion Sensor
"cc:a7:c1:6a:b5:78":1,# NEST Protect smoke alarm
"70:ee:50:96:bb:dc":1,# Netatmo Weather Station
"00:24:e4:e3:15:6e":1,# Withings Body+ (Scales)
"00:24:e4:e4:55:26":1,# Withings Body+ (Scales)
"00:24:e4:f6:91:38":1,# Withings Connect (Blood Pressure)
"00:24:e4:f7:ee:ac":1,# Withings Connect (Blood Pressure)
"70:3a:2d:4a:48:e2":0, # TUYA Smartdoor Bell
"b0:02:47:6f:63:37":2,# Pix-Star Easy Digital Photo Frame
"84:69:93:27:ad:35":2,# HP Envy
"18:48:be:31:4b:49":3,# Echo Show 8
"74:d4:23:32:a2:d7":3,# Echo Show 8
"6e:fe:2f:5a:d7:7e":1,# GALAXY Watch5 Pro
"90:48:6c:08:da:8a":0 # Ring Video Doorbell
}

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
test_file = "/scratch/apasquini/Data/Deakin/Deakin_com.npz"
train_file = "/scratch/apasquini/Data/UNSW/UNSW_com.npz"

phases = [("UNSW exact train with Deakin exact test", UNSW_Exact, Deakin_Exact),("UNSW close train with Deakin close test", UNSW_Close, Deakin_Close),
          ("UNSW full train with Deakin exact test", UNSW_Full, Deakin_Exact),("UNSW exact full with Deakin close test", UNSW_Full, Deakin_Close),
          ("UNSW meta exact train with Deakin meta exact test", UNSW_Exact_meta, Deakin_Exact_meta),("UNSW meta close train with Deakin meta close test", UNSW_Close_meta, Deakin_Close_meta),
          ("UNSW meta full train with Deakin meta exact test", UNSW_Full_meta, Deakin_Exact_meta),("UNSW meta full train with Deakin meta close test", UNSW_Full_meta, Deakin_Close_meta)]
for (phase, Uset, Dset) in phases:
    print(phase)
    print("==============================================")

    data = np.load(train_file)
    
    train=data['inputs']
    labels=data['labels']
    mac=data['macs']

    filtered_train = []
    filtered_labels = []

    mac_counts = defaultdict(int)

    for inp, label, m in zip(train, labels, mac):
        if m in Uset and mac_counts[m] < 100000:
            filtered_train.append(inp)
            filtered_labels.append(Uset[m])
            mac_counts[m] += 1

    del data

    num_labels =  max(Uset.values())+1
    
    # Create a distribution strategy
    rf = RandomForestClassifier(
        n_estimators=100,      # number of trees
        max_depth=None,        
        random_state=42,
        n_jobs=-1
        )
    rf.fit(filtered_train, filtered_labels)

    gb = HistGradientBoostingClassifier().fit(filtered_train, filtered_labels)

    data = np.load(test_file)


        

            
    test=data['inputs']
    labels=data['labels']
    mac=data['macs']
    filtered_test = []
    filtered_labels = []

    mac_counts = defaultdict(int)

    for inp, label, m in zip(test, labels, mac):
        if m in Dset and mac_counts[m] < 100000:
            filtered_test.append(inp)
            filtered_labels.append(Dset[m])
            mac_counts[m] += 1

    del data
    

    # Evaluate the model on the testing dataset
    rf_preds = rf.predict(filtered_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(filtered_labels, rf_preds)
    print("Random Forest Accuracy:", rf_accuracy)

    # Print classification report
    print("Random Forest Classification Report:")
    print(classification_report(filtered_labels, rf_preds))

    # Evaluate the model on the testing dataset
    gb_preds = gb.predict(filtered_test)

    # Calculate accuracy
    gb_accuracy = accuracy_score(filtered_labels, gb_preds)
    print("Gradient Boost Accuracy:", gb_accuracy)

    # Print classification report
    print("Gradient Boost Classification Report:")
    print(classification_report(filtered_labels, gb_preds))
