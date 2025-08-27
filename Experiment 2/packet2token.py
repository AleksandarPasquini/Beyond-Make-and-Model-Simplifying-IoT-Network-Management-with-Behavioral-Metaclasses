import warnings
import logging
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings("ignore", message="Calling str(pkt) on Python 3 makes no sense!")
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, Manager
import os
from scapy.all import *
import json

LSIF_mapping = {
"60:01:94:b6:1b:29":1,# Gosuna_Socket
"00:e0:4c:71:b6:b6":2, # Minger_LightStrip
"84:0d:8e:80:62:9a":2, # Smart_LightStrip
"84:20:96:90:43:e6":3, # itTiot_Camera
"14:6b:9c:85:8a:fe":3, # Tenvis_Camera
"24:f5:a2:ff:ad:97":1,# Wemo_SmartPlug
"84:f3:eb:3c:9e:4f":6,# LaCrosse_AlarmClock
"dc:4f:22:1b:2f:ad":2, # Lumiman_Bulb600
"34:03:de:14:dc:3b":3, # Ring_Doorbell
"2c:3a:e8:13:75:61":1,# oossxx_SmartPlug
"0c:80:63:65:64:de":1, # tp-link_SmartPlug
"bc:dd:c2:e1:25:29":2,  # Smart_Lamp
"50:c7:bf:eb:ec:c4":2,  # tp-link_LightBulb
"b2:c5:54:42:79:f2":3,  # D-Link_Camera936L
"28:ad:3e:72:bd:7e":3,  # Wans_Camera
"60:01:94:6e:3f:70":2,  # Lumiman_Bulb900
"84:0d:8e:40:04:54":1, # Lumiman_SmartPlug
"84:f3:eb:77:48:1f":2,  # Gosuna_LightBulb
"ec:3d:fd:5a:ba:79":6,  # Ocean_Radio
"60:01:94:a7:c9:fd":1, # Renpho_SmartPlug
"88:3f:4a:f5:de:d0":6,  # Chime_Doorbell
"84:f3:eb:18:91:dd":6 # Goumia_Coffeemaker
}

Aalto_mapping  = {
"00:17:88:24:76:ff":2, # Philips Hue Light Switch
"1c:5f:2b:aa:fd:4e":7, # D-Link Connected Home Hub
"94:10:3e:42:80:69":1, # WeMo Insight Switch 2
"3c:49:37:03:17:db":3, # Ednet Wireless indoor IP camera 1
"b0:c5:54:1c:71:85":3, # D-Link WiFi Day Camera
"94:10:3e:35:01:c1":1, # WeMo Switch model 1
"ac:cf:23:62:3c:6e":7, # Ednet living starter kit power Gateway
"00:24:e4:24:80:2a":5, # Withings Wireless Scale WS-30
"1c:5f:2b:aa:fd:4e":5, # D-Link Door & Window sensor
"74:da:38:80:79:fc":3, # Edimax IC-3115W Smart HD WiFi Network Camera 2
"b0:c5:54:25:5b:0e":3, # D-Link HD IP Camera
"94:10:3e:cd:37:65":7, # WeMo Link Lighting Bridge model
"50:c7:bf:00:c7:03":1, # TP-LinkPlugHS110
"3c:49:37:03:17:f0":3, # Ednet Wireless indoor IP camera 2
"5c:cf:7f:06:d9:02":6, # Smarter iKettle 2.0 water kettle 
"50:c7:bf:00:fc:a3":1, # TP-LinkPlugHS100
"20:f8:5e:ca:91:52":5, # Fitbit Aria WiFi-enabled scale
"5c:cf:7f:07:ae:fb":6, # Smarter SmarterCoffee coffee machine
"00:1a:22:05:c4:2e":1, # Homematic pluggable switch HMIP-PS
"00:1a:22:03:cb:be":7, # MAX! Cube LAN Gateway for MAX! Home automation sensors 
"84:18:26:7b:5f:6b":7, # Osram Lightify Gateway
"6c:72:20:c5:17:5a":5, # D-LinkWaterSensor
"74:da:38:80:7a:08":3,  # Edimax IC-3115W Smart HD WiFi Network Camera 1
"90:8d:78:a8:e1:43":5,  # D-Link WiFi Motion sensor
"74:da:38:4a:76:49":1,  # EdimaxPlug1101W
"94:10:3e:41:c2:05":1,  # WeMo Insight Switch 1
"90:8d:78:dd:0d:60":5,  # D-Link Siren
"94:10:3e:34:0c:b5":1,  # WeMo Switch model 2
"74:da:38:23:22:7b":1,  # EdimaxPlug2101W
"00:17:88:24:76:ff":7,  # Philips Hue Bridge
"90:8d:78:a8:e1:43":5   # "D-Link WiFi Motion sensor"
}

CIC_mapping = {
"1c:fe:2b:98:16:dd":4,# Amazon Alexa Echo Dot 1
"a0:d0:dc:c4:08:ff":4,# Amazon Alexa Echo Dot 2
"1c:12:b0:9b:0c:ec":4,# Amazon Alexa Echo Spot
"08:7c:39:ce:6e:2a":4,# Amazon Alexa Echo Studio
"cc:f4:11:9c:d0:00":4,# Google Nest Mini
"48:a6:b8:f9:1b:88":4,# Sonos One Speaker
"9c:8e:cd:1d:ab:9f":3,# AMCREST WiFi Camera
"3c:37:86:6f:b9:51":7,# Arlo Base Station
"40:5d:82:35:14:c8":3,# Arlo Q Camera
"c0:e7:bf:0a:79:d1":3,# Borun/Sichuan-AI Camera
"b0:c5:54:59:2e:99":3, # DCS8000LHA1 D-Link Mini Camera
"44:01:bb:ec:10:4a":3, # HeimVision Smart WiFi Camera
"34:75:63:73:f3:36":3, # Home Eye Camera
"7c:a7:b0:cd:18:32":3, # Luohe Cam Dog
"44:bb:3b:00:39:07":3, # Nest Indoor Camera
"70:ee:50:68:0e:32":3, # Netatmo Camera
"10:2c:6b:1b:43:be":3, # SIMCAM 1S (AMPAKTec)
"b8:5f:98:d0:76:e6":1,# Amazon Plug
"68:57:2d:56:ac:47":6,# Atomi Coffee Maker
"8c:85:80:6c:b6:47":7, # Eufy HomeBase 2
"50:02:91:b1:68:0c":2, # Globe Lamp ESP_B1680C
"b8:f0:09:03:9a:af":1,# Gosund ESP_039AAF Socket
"b8:f0:09:03:29:79":1,# Gosund ESP_032979 Plug
"50:02:91:10:09:8f":1,# Gosund ESP_10098F Socket
"c4:dd:57:0c:39:94":1,# Gosund ESP_0C3994 Plug
"50:02:91:1a:ce:e1":1,# Gosund ESP_1ACEE1 Socket
"24:a1:60:14:7f:f9":1,# Gosund ESP_147FF9 Plug
"50:02:91:10:ac:d8":1,# Gosund ESP_10ACD8 Plug
"d4:a6:51:30:64:b7":2, # HeimVision SmartLife Radio/Lamp
"00:17:88:60:d6:4f":7, # Philips Hue Bridge
"b0:09:da:3e:82:6c":7, # Ring Base Station AC:1236
"50:14:79:37:80:18":6,# iRobot Roomba
"00:02:75:f6:e3:cb":6,# Smart Board
"d4:a6:51:76:06:64":1,# Teckin Plug 1
"d4:a6:51:78:97:4e":1,# Teckin Plug 2
"d4:a6:51:20:91:d1":1,# Yutron Plug 1
"d4:a6:51:21:6c:29":1,# Yutron Plug 2
"f0:b4:d2:f9:60:95":5, # D-Link DCHS-161 Water Sensor
"ac:f1:08:4e:00:82":6,# LG Smart TV
"70:ee:50:6b:a8:1a":5 # Netatmo Weather Station
}

UNSW_mapping = {
"d0:52:a8:00:67:5e":7, # Smart Things
"44:65:0d:56:cc:d3":4, # Amazon Echo
"70:ee:50:18:34:43":3, # Netatmo Welcome
"f4:f2:6d:93:51:f1":3, # TP-Link Day Night Cloud camera
"00:16:6c:ab:6b:88":3, # Samsung SmartCam
"30:8c:fb:2f:e4:b2":3, # Dropcam
"00:62:6e:51:27:2e":3, # Insteon Camera
"e8:ab:fa:19:de:4f":3, # Insteon Camera
"00:24:e4:11:18:a8":3, # Withings Smart Baby Monitor
"ec:1a:59:79:f4:89":1, # Belkin Wemo switch
"50:c7:bf:00:56:39":1, # TP-Link Smart plug
"74:c6:3b:29:d7:1d":1, # iHome
"ec:1a:59:83:28:11":5,  # Belkin wemo motion sensor
"18:b4:30:25:be:e4":5,  # NEST Protect smoke alarm
"70:ee:50:03:b8:ac":5,  # Netatmo weather station
"00:24:e4:1b:6f:96":5, # Withings Smart scale
"74:6a:89:00:2e:25":5, # Blipcare Blood Pressure meter
"00:24:e4:20:28:c6":5, # Withings Aura smart sleep sensor
"d0:73:d5:01:83:08":2,  # Light Bulbs LiFX Smart Bulb
"18:b7:9e:02:20:44":4,  # Triby Speaker
"e0:76:d0:33:bb:85":6, # PIX-STAR Photo-frame
"70:5a:0f:e4:9b:c0":6, # HP Printer
"30:8c:fb:b6:ea:45":3  # Nest Dropcam
}

Deakin_mapping = {
"40:f6:bc:bc:89:7b":4,# Echo Dot (4th Gen)
"68:3a:48:0d:d4:1c":7,# Aeotec Smart Hub
"70:ee:50:57:95:29":3,# Netatmo Smart Indoor Security Camera
"54:af:97:bb:8d:8f":3,# TP-Link Tapo Pan/Tilt Wi-Fi Camera
"70:09:71:9d:ad:10":6, #32' Smart Monitor M80B UHD
"00:16:6c:d7:d5:f9":3, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
"40:ac:bf:29:04:d4":3, # EZVIZ Security Camera
"10:5a:17:b8:a2:0b":1, #TOPERSUN Smart Plug
"10:5a:17:b8:9f:70":1, #TOPERSUN Smart Plug
"fc:67:1f:53:fa:6e":5, # Perfk Motion Sensor
"1c:90:ff:bf:89:46":5,# Perfk Motion Sensor
"cc:a7:c1:6a:b5:78":5,# NEST Protect smoke alarm
"70:ee:50:96:bb:dc":5,# Netatmo Weather Station
"00:24:e4:e3:15:6e":5,# Withings Body+ (Scales)
"00:24:e4:e4:55:26":5,# Withings Body+ (Scales)
"00:24:e4:f6:91:38":5,# Withings Connect (Blood Pressure)
"00:24:e4:f7:ee:ac":5,# Withings Connect (Blood Pressure)
"70:3a:2d:4a:48:e2":3, # TUYA Smartdoor Bell
"b0:02:47:6f:63:37":6,# Pix-Star Easy Digital Photo Frame
"84:69:93:27:ad:35":6,# HP Envy
"18:48:be:31:4b:49":4,# Echo Show 8
"74:d4:23:32:a2:d7":4,# Echo Show 8
"6e:fe:2f:5a:d7:7e":5,# GALAXY Watch5 Pro
"90:48:6c:08:da:8a":3 # Ring Video Doorbell
}

def find_pcap_files(directory):
    pcap_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def packet_to_json(packet):
    """
    Convert a Scapy packet into a JSON-serializable dictionary.
    """
    packet_dict = {}
    
    def extract_fields(layer):
        
        fields = {}
        for field_name, field_value in layer.fields.items():
            # Mask sensitive fields
            if field_name in ['src', 'dst']:
                continue
            else:
                # Convert bytes to hex for readability
                if isinstance(field_value, bytes):
                    field_value = field_value.hex()
                # Handle lists (e.g., options in IP header)
                elif isinstance(field_value, list):
                    field_value = [str(item) for item in field_value]
                # Convert non-serializable types to strings
                elif not isinstance(field_value, (int, float, str, bool, type(None))):
                    field_value = str(field_value)
            fields[field_name] = field_value
        return {layer.name: fields}
    
    # Extract all layers
    layers = []
    current_layer = packet.payload
    while current_layer:
        layer_info = extract_fields(current_layer)
        layers.append(layer_info)
        current_layer = current_layer.payload if current_layer.payload else None
    
    packet_dict['layers'] = layers
    return packet_dict

mac_counts = None

def init_worker(shared_dict):
    """
    Called once in each worker process. Sets the global 'mac_counts' 
    to the provided Manager dictionary.
    """
    global mac_counts
    mac_counts = shared_dict

def process_pcap_file(args):
    """
    Process a PCAP file and tokenize each packet into GPT-2 tokens, capped at 
    100,000 packets per MAC address (SHARED across all processes, no lock).
    """
    # Suppress specific warnings and set logging level
    warnings.filterwarnings("ignore", message="Calling str(pkt) on Python 3 makes no sense!")
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

    pcap_file, MAC_addresses, max_length = args

    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    tokenized_packets = []  # To store tokenized output

    with PcapReader(pcap_file) as pcap_reader:
        for pkt in pcap_reader:
            # If cooked linux, re-build as Ether
            if pkt.haslayer("cooked linux"):
                pkt = Ether(bytes(pkt)[:12] + bytes(pkt)[14:])

            mac = pkt.src.lower() if pkt.src else None
            if not mac:
                continue

            # Check if MAC is in mapping
            if mac not in MAC_addresses or not pkt.haslayer(IP) :
                continue


            # -----------------------------------------
            # Shared dictionary logic (no lock)
            # -----------------------------------------
            current_count = mac_counts.get(mac, 0)
            if current_count >= 100000:
                # Already at or over 100k, skip
                continue
            # Increment global MAC count
            mac_counts[mac] = current_count + 1

            # Prepare label
            label = MAC_addresses[mac] - 1

            # Convert packet to JSON, then tokenize
            packet_json = packet_to_json(pkt)
            text_repr = json.dumps(packet_json)

            encoding = tokenizer.encode_plus(
                text_repr,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='tf'
            )

            # Convert to plain Python lists
            input_ids = encoding['input_ids'].numpy().tolist()[0]
            attention_mask = encoding['attention_mask'].numpy().tolist()[0]

            tokenized_packet = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'mac': mac
            }
            tokenized_packets.append(tokenized_packet)

    return tokenized_packets

def main():
    work = [
        ("/home-old/apasquini/Final_Exp/Data/Deakin/Deakin_1tokens.json", "/home-old/apasquini/Final_Exp/Data/Deakin", Deakin_mapping),
        ("/home-old/apasquini/Final_Exp/Data/Aalto/Aalto_1tokens.json", "/home-old/apasquini/Final_Exp/Data/Aalto", Aalto_mapping),
        ("/home-old/apasquini/Final_Exp/Data/UNSW/UNSW_1tokens.json", "/home-old/apasquini/Final_Exp/Data/UNSW", UNSW_mapping),
        ("/home-old/apasquini/Final_Exp/Data/LSIF/LSIF_1tokens.json", "/home-old/apasquini/Final_Exp/Data/LSIF", LSIF_mapping),
        ("/home-old/apasquini/Final_Exp/Data/CIC/CIC_1tokens.json", "/home-old/apasquini/Final_Exp/Data/CIC", CIC_mapping),
    ]

    for args_item in work:
        max_length = 1000
        tokens_output_file = args_item[0]
        directory_path = args_item[1]
        MAC_addresses = args_item[2]

        # Find PCAP files
        pcap_files = find_pcap_files(directory_path)
        args_list = [(pcap_file, MAC_addresses, max_length) for pcap_file in pcap_files]

        # ------------------------------------------------------
        # Create a Manager dict for shared counters (no lock).
        # ------------------------------------------------------
        manager = Manager()
        shared_mac_counts = manager.dict()

        # Optionally pre-initialize known MACs to 0:
        for mac in MAC_addresses:     
            shared_mac_counts[mac] = 0

        # Use a Pool with our 'init_worker' function
        with Pool(initializer=init_worker, initargs=(shared_mac_counts,)) as pool, open(tokens_output_file, 'w') as f:
            for tokenized_packets_per_file in tqdm(pool.imap(process_pcap_file, args_list), total=len(args_list)):
                # Write each tokenized packet to file
                for sequence in tokenized_packets_per_file:
                    json_line = json.dumps(sequence)
                    f.write(json_line + '\n')

        print(f"Tokenized packets saved to {tokens_output_file}")

if __name__ == '__main__':
    main()