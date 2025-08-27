import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


claude1 = {
    "Smart Cameras & Doorbells": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114, 121, 126],
    "Smart Plugs/Sockets/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    "Smart Speakers/Displays": [49, 50, 51, 52, 53, 76, 87, 104, 108, 124], 
    "Smart Home Hubs/Gateways": [24, 29, 40, 55, 67, 78, 86, 109], 
    "Smart Appliances": [7, 19, 22, 35, 38, 66, 79, 80, 84, 97, 105, 106, 112, 116, 122, 123, 125],  
    "Smart Health/Monitoring Devices": [30, 37, 42, 44, 46, 83, 85, 98, 99, 100, 101, 102, 117, 118, 119, 120, 31]
} # noticed that it created 8 categories intead of 7. Merged Doorbell and cameras together
missing1 = [31]
claude2 = {
    "Smart Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114],
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    "Smart Plugs & Sockets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Speakers & Displays": [19, 49, 50, 51, 52, 53, 76, 87, 104, 108, 124],
    "Smart Home Security & Sensors": [9, 21, 31, 42, 44, 46, 67, 78, 97, 98, 116, 117, 121, 126],
    "Health & Wellness Devices": [30, 37, 100, 101, 102, 119, 120, 125],
    "Smart Home Appliances & Hubs": [7, 22, 24, 29, 35, 38, 40, 66, 79, 80, 83, 84, 85, 86, 99, 105, 106, 109, 112, 118, 122, 123]
}
claude3 = {
    "Security Devices": [
        4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 78, 88, 89, 90, 91, 92, 93, 97, 107, 110, 111, 113, 114, 116, 121, 126
    ],
    "Smart Plugs/Switches": [
        1, 6, 10, 11, 17, 20, 23, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115
    ],
    "Lighting Devices": [
        2, 3, 8, 12, 13, 16, 18, 33, 41, 48, 68, 77, 103
    ],
    "Smart Hubs/Controllers": [
        24, 29, 40, 49, 50, 51, 52, 53, 76, 86, 87, 104, 109, 124, 108
    ],
    "Health & Wellness Devices": [
        30, 37, 100, 101, 102, 119, 120, 125
    ],
    "Home Appliances": [
        7, 19, 22, 35, 38, 66, 79, 84, 106, 112, 123, 80
    ],
    "Environmental Sensors": [
        42, 44, 83, 85, 98, 99, 105, 117, 118, 122
    ]
}
missing3 =[80, 108]
duplicate3 = [44]
claude4 = {
    "Security & Monitoring": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 78, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114, 116, 121, 126],
    
    "Lighting & Illumination": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 76, 77, 103],
    
    "Power Management": [1, 6, 10, 11, 17, 20, 25, 28, 29, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    
    "Smart Home Hubs & Assistants": [24, 40, 49, 50, 51, 52, 53, 67, 86, 87, 104, 108, 109, 124],
    
    "Health & Wellness": [30, 37, 100, 101, 119, 120, 125],
    
    "Appliances & Automation": [7, 19, 22, 35, 38, 66, 79, 80, 98, 106, 117, 123, 84, 105],
    
    "Environmental Monitoring": [42, 44, 83, 85, 97, 99, 102, 112, 118, 122]
}
missing4 = [84, 105]
duplicate4 = [44]
claude5 = {
    "Smart Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114],
    "Smart Plugs/Sockets/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    "Smart Home Hubs/Speakers/Assistants": [24, 29, 40, 49, 50, 51, 52, 53, 67, 78, 86, 87, 104, 109, 124],
    "Health & Monitoring Devices": [30, 31, 37, 42, 44, 46, 83, 85, 97, 98, 99, 100, 101, 102, 116, 117, 118, 119, 120, 125],
    "Smart Kitchen/Appliances": [7, 19, 22, 35, 38, 66, 76, 79],
    "Miscellaneous Smart Devices": [9, 21, 80, 84, 105, 106, 108, 112, 121, 122, 123, 126]
}
claude6 = {
    "Smart Lighting": [
        2,  # Minger_LightStrip
        3,  # Smart_LightStrip
        8,  # Lumiman_Bulb600
        12, # Smart_Lamp
        13, # tp-link_LightBulb
        16, # Lumiman_Bulb900
        18, # Gosuna_LightBulb
        23, # Philips Hue Light Switch
        33, # WeMo Link Lighting Bridge model
        41, # Osram Lightify Gateway
        48, # Philips Hue Bridge
        68, # Globe Lamp ESP_B1680C
        77, # Philips Hue Bridge (duplicate)
        103 # Light Bulbs LiFX Smart Bulb
    ],
    
    "Smart Plugs & Sockets": [
        1,  # Gosuna_Socket
        6,  # Wemo_SmartPlug
        10, # oossxx_SmartPlug
        11, # tp-link_SmartPlug
        17, # Lumiman_SmartPlug
        20, # Renpho_SmartPlug
        25, # WeMo Insight Switch
        28, # WeMo Switch model
        34, # TP-LinkPlugHS110
        36, # TP-LinkPlugHS100
        39, # Homematic pluggable switch HMIP-PS
        45, # EdimaxPlug1101W
        47, # EdimaxPlug2101W
        65, # Amazon Plug
        69, # Gosund ESP_039AAF Socket
        70, # Gosund ESP_032979 Plug
        71, # Gosund ESP_10098F Socket
        72, # Gosund ESP_0C3994 Plug
        73, # Gosund ESP_1ACEE1 Socket
        74, # Gosund ESP_147FF9 Plug
        75, # Gosund ESP_10ACD8 Plug
        81, # Teckin Plug
        82, # Yutron Plug
        94, # Belkin Wemo switch
        95, # TP-Link Smart plug
        96, # iHome Plug
        115 # TOPERSUN Smart Plug
    ],
    
    "Security & Surveillance": [
        4,  # itTiot_Camera
        5,  # Tenvis_Camera
        9,  # Ring_Doorbell
        14, # D-Link_Camera936L
        15, # Wans_Camera
        21, # Chime_Doorbell
        26, # Ednet Wireless indoor IP camera
        27, # D-Link WiFi Day Camera
        32, # D-Link HD IP Camera
        43, # Edimax IC-3115W Smart HD WiFi Network Camera
        54, # AMCREST WiFi Camera
        56, # Arlo Q Camera
        57, # Borun/Sichuan-AI Camera
        58, # DCS8000LHA1 D-Link Mini Camera
        59, # HeimVision Smart WiFi Camera
        60, # Home Eye Camera
        61, # Luohe Cam Dog
        62, # Nest Indoor Camera
        63, # Netatmo Camera
        64, # SIMCAM 1S (AMPAKTec)
        88, # Netatmo Welcome
        89, # TP-Link Day Night Cloud camera
        90, # Samsung SmartCam
        91, # Dropcam
        92, # Insteon Camera
        93, # Withings Smart Baby Monitor
        107, # Nest Dropcam
        110, # Netatmo Smart Indoor Security Camera
        111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
        113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
        114, # EZVIZ Security Camera
        121, # TUYA Smartdoor Bell
        126  # Ring Video Doorbell
    ],
    
    "Smart Hubs & Assistants": [
        24, # D-Link Connected Home Hub
        29, # Ednet living starter kit power Gateway
        40, # MAX! Cube LAN Gateway for MAX! Home automation sensors
        49, # Amazon Alexa Echo Dot
        50, # Amazon Alexa Echo Spot
        51, # Amazon Alexa Echo Studio
        52, # Google Nest Mini
        53, # Sonos One Speaker
        55, # Arlo Base Station
        67, # Eufy HomeBase
        78, # Ring Base Station AC:1236
        86, # Smart Things
        87, # Amazon Echo
        104, # Triby Speaker
        108, # Amazon Echo Dot (4th Gen)
        109, # Aeotec Smart Hub
        124  # Amazon Echo Show 8
    ],
    
    "Sensors & Monitoring Devices": [
        7,  # LaCrosse_AlarmClock
        19, # Ocean_Radio
        31, # D-Link Door & Window sensor
        42, # D-LinkWaterSensor
        44, # D-Link WiFi Motion sensor
        46, # D-Link Siren
        76, # HeimVision SmartLife Radio/Lamp
        83, # D-Link DCHS-161 Water Sensor
        85, # Netatmo Weather Station
        97, # Belkin wemo motion sensor
        98, # NEST Protect smoke alarm
        99, # Netatmo weather station
        102, # Withings Aura smart sleep sensor
        116, # Perfk Motion Sensor
        117, # NEST Protect smoke alarm (duplicate)
        118  # Netatmo Weather Station (duplicate)
    ],
    
    "Health & Wellness": [
        30, # Withings Wireless Scale WS-30
        37, # Fitbit Aria WiFi-enabled scale
        100, # Withings Smart scale
        101, # Blipcare Blood Pressure meter
        119, # Withings Body+ (Scales)
        120, # Withings Connect (Blood Pressure)
        125  # GALAXY Watch5 Pro
    ],
    
    "Smart Appliances & Other": [
        22, # Goumia_Coffeemaker
        35, # Smarter iKettle 2.0 water kettle
        38, # Smarter SmarterCoffee coffee machine
        66, # Atomi Coffee Maker
        79, # iRobot Roomba
        80, # Smart Board
        84, # LG Smart TV
        105, # PIX-STAR Photo-frame
        106, # HP Printer
        112, # 32' Smart Monitor M80B UHD
        122, # Pix-Star Easy Digital Photo Frame
        123  # HP Envy
    ]
}
claude7 = {
    "Smart Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114],
    
    "Smart Plugs/Sockets/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    
    "Smart Hubs/Gateways/Assistants": [24, 29, 40, 49, 50, 51, 52, 53, 67, 78, 86, 87, 104, 109, 124, 108],
    
    "Smart Sensors/Alarms": [7, 31, 42, 44, 46, 97, 98, 116, 117],
    
    "Smart Home Appliances": [22, 35, 38, 66, 79, 80, 106, 123],
    
    "Smart Health & Monitoring": [19, 30, 37, 76, 83, 84, 85, 99, 100, 101, 102, 105, 112, 118, 119, 120, 121, 122, 125, 126, 9, 21]
}
missing7 = [108]
duplicate7 = [35]
claude8 = {
    "Smart Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114],
    
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    
    "Smart Plugs/Sockets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    
    "Smart Sensors & Gateways": [24, 29, 31, 40, 42, 44, 46, 55, 67, 78, 83, 86, 97, 98, 109, 116, 117],
    
    "Smart Health & Monitoring": [30, 37, 85, 99, 100, 101, 102, 118, 119, 120, 125],
    
    "Smart Speakers & Entertainment": [19, 49, 50, 51, 52, 53, 76, 84, 87, 104, 105, 106, 108, 112, 122, 123, 124],
    
    "Smart Appliances & Other": [7, 9, 21, 22, 35, 38, 66, 79, 80, 121, 126]
}
claude9 = {
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    "Security & Monitoring": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 78, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114, 121, 126],
    "Smart Plugs & Outlets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Home Automation Hubs": [24, 29, 40, 49, 50, 51, 52, 53, 76, 86, 87, 104, 109, 124, 108],
    "Smart Appliances": [7, 19, 22, 35, 38, 66, 79, 80, 84, 106, 112, 123],
    "Health & Wellness": [30, 37, 100, 101, 102, 119, 120, 125],
    "Environmental Sensors": [42, 44, 83, 85, 97, 98, 99, 116, 117, 118, 122, 105]
}
missing9 = [108, 105]
duplicate9 = [44]
claude10 = {
    "Smart Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114],
    
    "Smart Plugs/Sockets/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    
    "Smart Lighting": [2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 77, 103],
    
    "Smart Hubs/Gateways": [24, 29, 40, 67, 86, 109],
    
    "Smart Speakers/Displays": [19, 49, 50, 51, 52, 53, 76, 87, 104, 108, 124],
    
    "Smart Health/Monitoring": [30, 37, 42, 44, 46, 83, 85, 97, 98, 99, 100, 101, 102, 116, 117, 118, 119, 120, 125],
    
    "Smart Appliances/Other": [7, 9, 21, 22, 31, 35, 38, 66, 78, 79, 80, 84, 105, 106, 112, 121, 122, 123, 126]
}

runs = [claude1, claude2, claude3, claude4, claude5, claude6, claude7, claude8, claude9, claude10]
for x in runs:
    s = set()
    dup = []
    flat = [z for y in x.values() for z in y]
    if len(x.keys()) != 7:
        print("Error")
    if len(flat) >126:
        for n in flat:
          if n in s:
              dup.append(n)
          else:
              s.add(n)
        print("Error")
        print(dup)
        print(len(flat))
    if len(flat) <126:
      missing = []
      for a in range(1, 126):
          if a not in flat:
              missing.append(a)
      print(missing)
    print(list(x.keys()))

concepts = {
    1: [
        "Smart Cameras & Doorbells", 
        "Smart Cameras", 
        "Security Devices", 
        "Security & Monitoring", 
        "Smart Cameras", 
        "Security & Surveillance", 
        "Smart Cameras", 
        "Smart Cameras", 
        "Security & Monitoring", 
        "Smart Cameras"
    ],
    
    2: [
        "Smart Plugs/Sockets/Switches", 
        "Smart Plugs & Sockets", 
        "Smart Plugs/Switches", 
        "Power Management", 
        "Smart Plugs/Sockets/Switches", 
        "Smart Plugs & Sockets", 
        "Smart Plugs/Sockets/Switches", 
        "Smart Plugs/Sockets", 
        "Smart Plugs & Outlets", 
        "Smart Plugs/Sockets/Switches"
    ],
    
    3: [
        "Smart Lighting", 
        "Smart Lighting", 
        "Lighting Devices", 
        "Lighting & Illumination", 
        "Smart Lighting", 
        "Smart Lighting", 
        "Smart Lighting", 
        "Smart Lighting", 
        "Smart Lighting", 
        "Smart Lighting"
    ],
    
    4: [
        "Smart Home Hubs/Gateways", 
        "Smart Speakers & Displays", 
        "Smart Hubs/Controllers", 
        "Smart Home Hubs & Assistants", 
        "Smart Home Hubs/Speakers/Assistants", 
        "Smart Hubs & Assistants", 
        "Smart Hubs/Gateways/Assistants", 
        "Smart Sensors & Gateways", 
        "Home Automation Hubs", 
        "Smart Hubs/Gateways",
        "Smart Speakers/Displays", 
        "Smart Speakers & Displays", 
        "Smart Home Hubs/Speakers/Assistants", 
        "Smart Speakers & Entertainment", 
        "Smart Speakers/Displays"
    ],
    
    5: [
        "Smart Health/Monitoring Devices", 
        "Health & Wellness Devices", 
        "Health & Wellness Devices", 
        "Health & Wellness", 
        "Health & Monitoring Devices", 
        "Health & Wellness", 
        "Smart Health & Monitoring", 
        "Smart Health & Monitoring", 
        "Health & Wellness", 
        "Smart Health/Monitoring"
    ],
    
    6: [
        "Smart Appliances", 
        "Smart Home Appliances & Hubs", 
        "Home Appliances", 
        "Appliances & Automation", 
        "Smart Kitchen/Appliances", 
        "Smart Appliances & Other", 
        "Smart Home Appliances", 
        "Smart Appliances & Other", 
        "Smart Appliances", 
        "Smart Appliances/Other",
        "Miscellaneous Smart Devices",
    ],
    
    7: [
        "Smart Home Security & Sensors", 
        "Environmental Sensors", 
        "Environmental Monitoring", 
        "Sensors & Monitoring Devices", 
        "Smart Sensors/Alarms", 
        "Environmental Sensors"
    ],
}
names = {
    "Security & Monitoring": 1,
    
    "Power Management": 2,
    
    "Lighting": 3,
    
    "Smart Hubs & Displays": 4,
    
    "Health & Wellness": 5,
    
    "Smart Appliances": 6,
    
    "Environmental Sensors": 7,
    
}


inverse_concepts = {name: key for key, names in concepts.items() for name in names}


data = []
for rater_id, device_dict in enumerate(runs, start=1):
    for name, subject in device_dict.items():
        score = inverse_concepts[name]
        for device in subject:
            data.append([device, rater_id, score])

# Convert to pandas DataFrame
df = pd.DataFrame(data, columns=['subject', 'rater', 'score'])
print(df)
count_matrix = (
    df.groupby(['subject', 'score'])['rater']
      .count()
      .unstack(level='score', fill_value=0)
)

print("\nFleiss count matrix (rows=subjects, columns=category labels):\n", count_matrix)

# 4) Compute Fleiss’ Kappa
kappa_value = fleiss_kappa(count_matrix.values)
print("\nFleiss’ Kappa:", kappa_value)