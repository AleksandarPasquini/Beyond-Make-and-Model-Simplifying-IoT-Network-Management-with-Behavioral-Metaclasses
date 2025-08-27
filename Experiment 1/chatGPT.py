import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

ChatGPT_1 = {
  "Security": [
    4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 42, 43, 44, 46,
    54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 83, 88, 89, 90,
    91, 92, 93, 97, 98, 107, 110, 111, 113, 114, 116, 117, 121, 126
  ],
  "Plugs & Switches": [
    1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65,
    69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115
  ],
  "Lighting": [
    2, 3, 8, 12, 13, 16, 18, 23, 68, 76, 103
  ],
  "Voice Assistants & Smart Speakers": [
    49, 50, 51, 52, 53, 87, 104, 108, 124
  ],
  "Health & Fitness": [
    30, 37, 100, 101, 102, 119, 120, 125
  ],
  "Hubs & Gateways": [
    24, 29, 33, 40, 41, 48, 55, 67, 77, 78, 86, 109
  ],
  "Appliances & Others": [
    7, 19, 22, 35, 38, 66, 79, 80, 84, 85, 99, 105,
    106, 112, 118, 122, 123
  ]
}
ChatGPT_2 = {
    "Smart Plugs & Switches": [
        1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65,
        69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115
    ],
    "Smart Lighting": [
        2, 3, 8, 12, 13, 16, 18, 23, 68, 76, 103
    ],
    "Cameras": [
        4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 107, 110,
        111, 113, 114
    ],
    "Doorbells": [
        9, 21, 121, 126
    ],
    "Hubs/Gateways": [
        24, 29, 33, 40, 41, 48, 55, 67, 77, 78, 86, 109
    ],
    "Sensors & Alarms": [
        31, 42, 44, 46, 83, 85, 97, 98, 99, 102, 116, 117, 118
    ],
    "Others": [
        7, 19, 22, 30, 35, 37, 38, 49, 50, 51, 52, 53, 66, 79,
        80, 84, 87, 100, 101, 104, 105, 106, 108, 112, 119,
        120, 122, 123, 124, 125
    ]
}
duplicates_2 = [9, 126]
ChatGPT_3 = {
    "Cameras": [
        4,   # itTiot_Camera
        5,   # Tenvis_Camera
        9,   # Ring_Doorbell (video doorbell)
        14,  # D-Link_Camera936L
        15,  # Wans_Camera
        26,  # Ednet Wireless indoor IP camera
        27,  # D-Link WiFi Day Camera
        32,  # D-Link HD IP Camera
        43,  # Edimax IC-3115W Smart HD WiFi Network Camera
        54,  # AMCREST WiFi Camera
        56,  # Arlo Q Camera
        57,  # Borun/Sichuan-AI Camera
        58,  # DCS8000LHA1 D-Link Mini Camera
        59,  # HeimVision Smart WiFi Camera
        60,  # Home Eye Camera
        61,  # Luohe Cam Dog
        62,  # Nest Indoor Camera
        63,  # Netatmo Camera
        64,  # SIMCAM 1S (AMPAKTec)
        88,  # Netatmo Welcome
        89,  # TP-Link Day Night Cloud camera
        90,  # Samsung SmartCam
        91,  # Dropcam
        92,  # Insteon Camera
        93,  # Withings Smart Baby Monitor
        107, # Nest Dropcam
        110, # Netatmo Smart Indoor Security Camera
        111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
        113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
        114, # EZVIZ Security Camera
        121, # TUYA Smartdoor Bell (video doorbell)
        126  # Ring Video Doorbell
    ],
    "Lights": [
        2,   # Minger_LightStrip
        3,   # Smart_LightStrip
        8,   # Lumiman_Bulb600
        12,  # Smart_Lamp
        13,  # tp-link_LightBulb
        16,  # Lumiman_Bulb900
        18,  # Gosuna_LightBulb
        23,  # Philips Hue Light Switch (treated as part of the lighting system)
        68,  # Globe Lamp ESP_B1680C
        103  # Light Bulbs LiFX Smart Bulb
    ],
    "Plugs/Switches": [
        1,   # Gosuna_Socket
        6,   # Wemo_SmartPlug
        10,  # oossxx_SmartPlug
        11,  # tp-link_SmartPlug
        17,  # Lumiman_SmartPlug
        20,  # Renpho_SmartPlug
        25,  # WeMo Insight Switch
        28,  # WeMo Switch model
        34,  # TP-LinkPlugHS110
        36,  # TP-LinkPlugHS100
        39,  # Homematic pluggable switch HMIP-PS
        45,  # EdimaxPlug1101W
        47,  # EdimaxPlug2101W
        65,  # Amazon Plug
        69,  # Gosund ESP_039AAF Socket
        70,  # Gosund ESP_032979 Plug
        71,  # Gosund ESP_10098F Socket
        72,  # Gosund ESP_0C3994 Plug
        73,  # Gosund ESP_1ACEE1 Socket
        74,  # Gosund ESP_147FF9 Plug
        75,  # Gosund ESP_10ACD8 Plug
        81,  # Teckin Plug
        82,  # Yutron Plug
        94,  # Belkin Wemo switch
        95,  # TP-Link Smart plug
        96,  # iHome Plug
        115  # TOPERSUN Smart Plug
    ],
    "Voice Assistants": [
        49,  # Amazon Alexa Echo Dot
        50,  # Amazon Alexa Echo Spot
        51,  # Amazon Alexa Echo Studio
        52,  # Google Nest Mini
        53,  # Sonos One Speaker
        87,  # Amazon Echo
        104, # Triby Speaker
        108, # Amazon Echo Dot (4th Gen)
        124  # Amazon Echo Show 8
    ],
    "Hubs/Gateways": [
        24,  # D-Link Connected Home Hub
        29,  # Ednet living starter kit power Gateway
        33,  # WeMo Link Lighting Bridge model
        40,  # MAX! Cube LAN Gateway for MAX! Home automation sensors
        41,  # Osram Lightify Gateway
        48,  # Philips Hue Bridge
        55,  # Arlo Base Station
        67,  # Eufy HomeBase
        77,  # Philips Hue Bridge (duplicate entry)
        78,  # Ring Base Station AC:1236
        86,  # Smart Things
        109  # Aeotec Smart Hub
    ],
    "Sensors": [
        31,  # D-Link Door & Window sensor
        42,  # D-LinkWaterSensor
        44,  # D-Link WiFi Motion sensor
        83,  # D-Link DCHS-161 Water Sensor
        85,  # Netatmo Weather Station
        97,  # Belkin wemo motion sensor
        98,  # NEST Protect smoke alarm
        99,  # Netatmo weather station
        102, # Withings Aura smart sleep sensor
        116, # Perfk Motion Sensor
        117, # NEST Protect smoke alarm (duplicate)
        118  # Netatmo Weather Station (duplicate)
    ],
    "Others": [
        7,   # LaCrosse_AlarmClock
        19,  # Ocean_Radio
        21,  # Chime_Doorbell (no camera)
        22,  # Goumia_Coffeemaker
        30,  # Withings Wireless Scale WS-30
        35,  # Smarter iKettle 2.0 water kettle
        37,  # Fitbit Aria WiFi-enabled scale
        38,  # Smarter SmarterCoffee coffee machine
        46,  # D-Link Siren
        66,  # Atomi Coffee Maker
        76,  # HeimVision SmartLife Radio/Lamp
        79,  # iRobot Roomba
        80,  # Smart Board
        84,  # LG Smart TV
        100, # Withings Smart scale
        101, # Blipcare Blood Pressure meter
        105, # PIX-STAR Photo-frame
        106, # HP Printer
        112, # 32' Smart Monitor M80B UHD
        119, # Withings Body+ (Scales)
        120, # Withings Connect (Blood Pressure)
        122, # Pix-Star Easy Digital Photo Frame
        123, # HP Envy
        125  # GALAXY Watch5 Pro
    ]
}
ChatGPT_4 = {
    "Cameras": [
        4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 88, 89, 90, 91, 92, 93, 107, 110, 111, 113, 114
    ],
    "Plugs & Switches": [
        1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72,
        73, 74, 75, 81, 82, 94, 95, 96, 115
    ],
    "Lights & Bulbs": [
        2, 3, 8, 12, 13, 16, 18, 23, 33, 41, 48, 68, 76, 77, 103
    ],
    "Voice Assistants & Speakers": [
        49, 50, 51, 52, 53, 87, 104, 108, 124
    ],
    "Doorbells": [
        9, 21, 78, 121, 126
    ],
    "Sensors & Alarms": [
        31, 42, 44, 46, 83, 85, 97, 98, 99, 116, 117, 118
    ],
    "Others": [
        7, 19, 22, 24, 29, 30, 35, 37, 38, 40, 66, 67, 79, 80, 84, 86, 100,
        101, 102, 105, 106, 109, 112, 119, 120, 122, 123, 125
    ]
}
ChatGPT_5 = {
    "Smart Plugs & Switches": [
        1, 6, 10, 11, 17, 20, 25, 28, 34, 36,
        39, 45, 47, 65, 69, 70, 71, 72, 73, 74,
        75, 81, 82, 94, 95, 96, 115
    ],
    "Smart Lighting": [
        2, 3, 8, 12, 13, 16, 18, 23, 68, 103
    ],
    "Cameras": [
        4, 5, 14, 15, 26, 27, 32, 43, 54, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90,
        91, 92, 93, 107, 110, 111, 113, 114
    ],
    "Doorbells & Security": [
        9, 21, 31, 42, 44, 46, 83, 97, 98, 116,
        117, 121, 126
    ],
    "Smart Speakers & Voice Assistants": [
        49, 50, 51, 52, 53, 87, 104, 108, 124
    ],
    "Hubs/Gateways/Bridges": [
        24, 29, 33, 40, 41, 48, 55, 67, 77, 78,
        86, 109
    ],
    "Home Appliances & Misc": [
        7, 19, 22, 30, 35, 37, 38, 66, 76, 79,
        80, 84, 85, 99, 100, 101, 102, 105, 106,
        112, 118, 119, 120, 122, 123, 125
    ]
}
duplicates_5= [9, 121, 126, 55, 78, 76]
ChatGPT_6 = {
    "Cameras & Doorbells": [
        4,   # itTiot_Camera
        5,   # Tenvis_Camera
        9,   # Ring_Doorbell
        14,  # D-Link_Camera936L
        15,  # Wans_Camera
        21,  # Chime_Doorbell
        26,  # Ednet Wireless indoor IP camera
        27,  # D-Link WiFi Day Camera
        32,  # D-Link HD IP Camera
        43,  # Edimax IC-3115W Smart HD WiFi Network Camera
        54,  # AMCREST WiFi Camera
        56,  # Arlo Q Camera
        57,  # Borun/Sichuan-AI Camera
        58,  # DCS8000LHA1 D-Link Mini Camera
        59,  # HeimVision Smart WiFi Camera
        60,  # Home Eye Camera
        61,  # Luohe Cam Dog
        62,  # Nest Indoor Camera
        63,  # Netatmo Camera
        64,  # SIMCAM 1S (AMPAKTec)
        88,  # Netatmo Welcome
        89,  # TP-Link Day Night Cloud camera
        90,  # Samsung SmartCam
        91,  # Dropcam
        92,  # Insteon Camera
        93,  # Withings Smart Baby Monitor (also in Health & Wearables)
        107, # Nest Dropcam
        110, # Netatmo Smart Indoor Security Camera
        111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
        113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
        114, # EZVIZ Security Camera
        121, # TUYA Smartdoor Bell
        126, # Ring Video Doorbell
    ],
    
    "Plugs & Switches": [
        1,   # Gosuna_Socket
        6,   # Wemo_SmartPlug
        10,  # oossxx_SmartPlug
        11,  # tp-link_SmartPlug
        17,  # Lumiman_SmartPlug
        20,  # Renpho_SmartPlug
        25,  # WeMo Insight Switch
        28,  # WeMo Switch model
        34,  # TP-LinkPlugHS110
        36,  # TP-LinkPlugHS100
        39,  # Homematic pluggable switch HMIP-PS
        45,  # EdimaxPlug1101W
        47,  # EdimaxPlug2101W
        65,  # Amazon Plug
        69,  # Gosund ESP_039AAF Socket
        70,  # Gosund ESP_032979 Plug
        71,  # Gosund ESP_10098F Socket
        72,  # Gosund ESP_0C3994 Plug
        73,  # Gosund ESP_1ACEE1 Socket
        74,  # Gosund ESP_147FF9 Plug
        75,  # Gosund ESP_10ACD8 Plug
        81,  # Teckin Plug
        82,  # Yutron Plug
        94,  # Belkin Wemo switch
        95,  # TP-Link Smart plug
        96,  # iHome Plug
        115, # TOPERSUN Smart Plug
    ],
    
    "Lighting": [
        2,   # Minger_LightStrip
        3,   # Smart_LightStrip
        8,   # Lumiman_Bulb600
        12,  # Smart_Lamp
        13,  # tp-link_LightBulb
        16,  # Lumiman_Bulb900
        18,  # Gosuna_LightBulb
        23,  # Philips Hue Light Switch
        68,  # Globe Lamp ESP_B1680C
        103, # Light Bulbs LiFX Smart Bulb
    ],
    
    "Voice Assistants & Smart Speakers": [
        49,  # Amazon Alexa Echo Dot
        50,  # Amazon Alexa Echo Spot
        51,  # Amazon Alexa Echo Studio
        52,  # Google Nest Mini
        53,  # Sonos One Speaker
        87,  # Amazon Echo
        104, # Triby Speaker
        108, # Amazon Echo Dot (4th Gen)
        124, # Amazon Echo Show 8
    ],
    
    "Hubs / Gateways / Security Sensors": [
        24,  # D-Link Connected Home Hub
        29,  # Ednet living starter kit power Gateway
        31,  # D-Link Door & Window sensor
        33,  # WeMo Link Lighting Bridge model
        40,  # MAX! Cube LAN Gateway for MAX! Home automation sensors
        41,  # Osram Lightify Gateway
        42,  # D-LinkWaterSensor
        44,  # D-Link WiFi Motion sensor
        46,  # D-Link Siren
        48,  # Philips Hue Bridge
        55,  # Arlo Base Station
        67,  # Eufy HomeBase
        77,  # Philips Hue Bridge
        78,  # Ring Base Station AC:1236
        83,  # D-Link DCHS-161 Water Sensor
        85,  # Netatmo Weather Station
        86,  # Smart Things
        97,  # Belkin wemo motion sensor
        98,  # NEST Protect smoke alarm
        99,  # Netatmo weather station
        109, # Aeotec Smart Hub
        116, # Perfk Motion Sensor
        117, # NEST Protect smoke alarm
        118, # Netatmo Weather Station
    ],
    
    "Health & Wearables": [
        30,  # Withings Wireless Scale WS-30
        37,  # Fitbit Aria WiFi-enabled scale
        100, # Withings Smart scale
        101, # Blipcare Blood Pressure meter
        102, # Withings Aura smart sleep sensor
        119, # Withings Body+ (Scales)
        120, # Withings Connect (Blood Pressure)
        125, # GALAXY Watch5 Pro
    ],
    
    "Smart Appliances & Others": [
        7,   # LaCrosse_AlarmClock
        19,  # Ocean_Radio
        22,  # Goumia_Coffeemaker
        35,  # Smarter iKettle 2.0 water kettle
        38,  # Smarter SmarterCoffee coffee machine
        66,  # Atomi Coffee Maker
        76,  # HeimVision SmartLife Radio/Lamp
        79,  # iRobot Roomba
        80,  # Smart Board
        84,  # LG Smart TV
        105, # PIX-STAR Photo-frame
        106, # HP Printer
        112, # 32' Smart Monitor M80B UHD
        122, # Pix-Star Easy Digital Photo Frame
        123, # HP Envy
    ],
}
duplicates_6 =[93]
ChatGPT_7 = {
  "Plugs and Switches": [
    1,   # Gosuna_Socket
    6,   # Wemo_SmartPlug
    10,  # oossxx_SmartPlug
    11,  # tp-link_SmartPlug
    17,  # Lumiman_SmartPlug
    20,  # Renpho_SmartPlug
    25,  # WeMo Insight Switch
    28,  # WeMo Switch model
    34,  # TP-LinkPlugHS110
    36,  # TP-LinkPlugHS100
    39,  # Homematic pluggable switch HMIP-PS
    45,  # EdimaxPlug1101W
    47,  # EdimaxPlug2101W
    65,  # Amazon Plug
    69,  # Gosund ESP_039AAF Socket
    70,  # Gosund ESP_032979 Plug
    71,  # Gosund ESP_10098F Socket
    72,  # Gosund ESP_0C3994 Plug
    73,  # Gosund ESP_1ACEE1 Socket
    74,  # Gosund ESP_147FF9 Plug
    75,  # Gosund ESP_10ACD8 Plug
    81,  # Teckin Plug
    82,  # Yutron Plug
    94,  # Belkin Wemo switch
    95,  # TP-Link Smart plug
    96,  # iHome Plug
    115  # TOPERSUN Smart Plug
  ],

  "Lighting": [
    2,   # Minger_LightStrip
    3,   # Smart_LightStrip
    8,   # Lumiman_Bulb600
    12,  # Smart_Lamp
    13,  # tp-link_LightBulb
    16,  # Lumiman_Bulb900
    18,  # Gosuna_LightBulb
    23,  # Philips Hue Light Switch
    68,  # Globe Lamp ESP_B1680C
    103  # Light Bulbs LiFX Smart Bulb
  ],

  "Cameras": [
    4,   # itTiot_Camera
    5,   # Tenvis_Camera
    14,  # D-Link_Camera936L
    15,  # Wans_Camera
    26,  # Ednet Wireless indoor IP camera
    27,  # D-Link WiFi Day Camera
    32,  # D-Link HD IP Camera
    43,  # Edimax IC-3115W Smart HD WiFi Network Camera
    54,  # AMCREST WiFi Camera
    56,  # Arlo Q Camera
    57,  # Borun/Sichuan-AI Camera
    58,  # DCS8000LHA1 D-Link Mini Camera
    59,  # HeimVision Smart WiFi Camera
    60,  # Home Eye Camera
    61,  # Luohe Cam Dog
    62,  # Nest Indoor Camera
    63,  # Netatmo Camera
    64,  # SIMCAM 1S (AMPAKTec)
    88,  # Netatmo Welcome
    89,  # TP-Link Day Night Cloud camera
    90,  # Samsung SmartCam
    91,  # Dropcam
    92,  # Insteon Camera
    93,  # Withings Smart Baby Monitor
    107, # Nest Dropcam
    110, # Netatmo Smart Indoor Security Camera
    111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
    113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
    114  # EZVIZ Security Camera
  ],

  "Voice Assistants & Speakers": [
    49,  # Amazon Alexa Echo Dot
    50,  # Amazon Alexa Echo Spot
    51,  # Amazon Alexa Echo Studio
    52,  # Google Nest Mini
    53,  # Sonos One Speaker
    87,  # Amazon Echo
    104, # Triby Speaker
    108, # Amazon Echo Dot (4th Gen)
    124  # Amazon Echo Show 8
  ],

  "Hubs / Gateways": [
    24,  # D-Link Connected Home Hub
    29,  # Ednet living starter kit power Gateway
    33,  # WeMo Link Lighting Bridge model
    40,  # MAX! Cube LAN Gateway for MAX! Home automation sensors
    41,  # Osram Lightify Gateway
    48,  # Philips Hue Bridge
    55,  # Arlo Base Station
    67,  # Eufy HomeBase
    77,  # Philips Hue Bridge (duplicate)
    78,  # Ring Base Station AC:1236
    86,  # Smart Things
    109  # Aeotec Smart Hub
  ],

  "Sensors, Alarms & Doorbells": [
    9,    # Ring_Doorbell
    21,   # Chime_Doorbell
    31,   # D-Link Door & Window sensor
    42,   # D-LinkWaterSensor
    44,   # D-Link WiFi Motion sensor
    46,   # D-Link Siren
    83,   # D-Link DCHS-161 Water Sensor
    85,   # Netatmo Weather Station
    97,   # Belkin wemo motion sensor
    98,   # NEST Protect smoke alarm
    99,   # Netatmo weather station
    116,  # Perfk Motion Sensor
    117,  # NEST Protect smoke alarm (duplicate)
    118,  # Netatmo Weather Station (duplicate)
    121,  # TUYA Smartdoor Bell
    126   # Ring Video Doorbell
  ],

  "Misc / Appliances": [
    7,    # LaCrosse_AlarmClock
    19,   # Ocean_Radio
    22,   # Goumia_Coffeemaker
    30,   # Withings Wireless Scale WS-30
    35,   # Smarter iKettle 2.0 water kettle
    37,   # Fitbit Aria WiFi-enabled scale
    38,   # Smarter SmarterCoffee coffee machine
    66,   # Atomi Coffee Maker
    76,   # HeimVision SmartLife Radio/Lamp
    79,   # iRobot Roomba
    80,   # Smart Board
    84,   # LG Smart TV
    100,  # Withings Smart scale
    101,  # Blipcare Blood Pressure meter
    102,  # Withings Aura smart sleep sensor
    105,  # PIX-STAR Photo-frame
    106,  # HP Printer
    112,  # 32' Smart Monitor M80B UHD
    119,  # Withings Body+ (Scales)
    120,  # Withings Connect (Blood Pressure)
    122,  # Pix-Star Easy Digital Photo Frame
    123,  # HP Envy
    125   # GALAXY Watch5 Pro
  ]
}
ChatGPT_8 = {
    "Cameras": [
        4,   # itTiot_Camera
        5,   # Tenvis_Camera
        14,  # D-Link_Camera936L
        15,  # Wans_Camera
        26,  # Ednet Wireless indoor IP camera
        27,  # D-Link WiFi Day Camera
        32,  # D-Link HD IP Camera
        43,  # Edimax IC-3115W Smart HD WiFi Network Camera
        54,  # AMCREST WiFi Camera
        56,  # Arlo Q Camera
        57,  # Borun/Sichuan-AI Camera
        58,  # DCS8000LHA1 D-Link Mini Camera
        59,  # HeimVision Smart WiFi Camera
        60,  # Home Eye Camera
        61,  # Luohe Cam Dog
        62,  # Nest Indoor Camera
        63,  # Netatmo Camera
        64,  # SIMCAM 1S (AMPAKTec)
        88,  # Netatmo Welcome
        89,  # TP-Link Day Night Cloud camera
        90,  # Samsung SmartCam
        91,  # Dropcam
        92,  # Insteon Camera
        93,  # Withings Smart Baby Monitor
        107, # Nest Dropcam
        110, # Netatmo Smart Indoor Security Camera
        111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
        113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
        114  # EZVIZ Security Camera
    ],

    "Plugs_Switches": [
        1,   # Gosuna_Socket
        6,   # Wemo_SmartPlug
        10,  # oossxx_SmartPlug
        11,  # tp-link_SmartPlug
        17,  # Lumiman_SmartPlug
        20,  # Renpho_SmartPlug
        25,  # WeMo Insight Switch
        28,  # WeMo Switch model
        34,  # TP-LinkPlugHS110
        36,  # TP-LinkPlugHS100
        39,  # Homematic pluggable switch HMIP-PS
        45,  # EdimaxPlug1101W
        47,  # EdimaxPlug2101W
        65,  # Amazon Plug
        69,  # Gosund ESP_039AAF Socket
        70,  # Gosund ESP_032979 Plug
        71,  # Gosund ESP_10098F Socket
        72,  # Gosund ESP_0C3994 Plug
        73,  # Gosund ESP_1ACEE1 Socket
        74,  # Gosund ESP_147FF9 Plug
        75,  # Gosund ESP_10ACD8 Plug
        81,  # Teckin Plug
        82,  # Yutron Plug
        94,  # Belkin Wemo switch
        95,  # TP-Link Smart plug
        96,  # iHome Plug
        115  # TOPERSUN Smart Plug
    ],

    "Lights": [
        2,   # Minger_LightStrip
        3,   # Smart_LightStrip
        8,   # Lumiman_Bulb600
        12,  # Smart_Lamp
        13,  # tp-link_LightBulb
        16,  # Lumiman_Bulb900
        18,  # Gosuna_LightBulb
        23,  # Philips Hue Light Switch
        68,  # Globe Lamp ESP_B1680C
        76,  # HeimVision SmartLife Radio/Lamp
        103  # Light Bulbs LiFX Smart Bulb
    ],

    "Hubs_Gateways": [
        24,  # D-Link Connected Home Hub
        29,  # Ednet living starter kit power Gateway
        33,  # WeMo Link Lighting Bridge model
        40,  # MAX! Cube LAN Gateway for MAX! Home automation sensors
        41,  # Osram Lightify Gateway
        48,  # Philips Hue Bridge
        55,  # Arlo Base Station
        67,  # Eufy HomeBase
        77,  # Philips Hue Bridge (duplicate listing)
        78,  # Ring Base Station AC:1236
        86,  # Smart Things (hub)
        109  # Aeotec Smart Hub
    ],

    "Voice_Assistants": [
        49,  # Amazon Alexa Echo Dot
        50,  # Amazon Alexa Echo Spot
        51,  # Amazon Alexa Echo Studio
        52,  # Google Nest Mini
        53,  # Sonos One Speaker
        87,  # Amazon Echo
        104, # Triby Speaker
        108, # Amazon Echo Dot (4th Gen)
        124  # Amazon Echo Show 8
    ],

    "Sensors_Security": [
        9,   # Ring_Doorbell
        21,  # Chime_Doorbell
        31,  # D-Link Door & Window sensor
        42,  # D-LinkWaterSensor
        44,  # D-Link WiFi Motion sensor
        46,  # D-Link Siren
        83,  # D-Link DCHS-161 Water Sensor
        85,  # Netatmo Weather Station (duplicate listing)
        97,  # Belkin wemo motion sensor
        98,  # NEST Protect smoke alarm
        99,  # Netatmo weather station
        116, # Perfk Motion Sensor
        117, # NEST Protect smoke alarm (duplicate listing)
        118, # Netatmo Weather Station (duplicate listing)
        121, # TUYA Smartdoor Bell
        126  # Ring Video Doorbell
    ],

    "Appliances_Other": [
        7,   # LaCrosse_AlarmClock
        19,  # Ocean_Radio
        22,  # Goumia_Coffeemaker
        30,  # Withings Wireless Scale WS-30
        35,  # Smarter iKettle 2.0 water kettle
        37,  # Fitbit Aria WiFi-enabled scale
        38,  # Smarter SmarterCoffee coffee machine
        66,  # Atomi Coffee Maker
        79,  # iRobot Roomba
        80,  # Smart Board
        84,  # LG Smart TV
        100, # Withings Smart scale
        101, # Blipcare Blood Pressure meter
        102, # Withings Aura smart sleep sensor
        105, # PIX-STAR Photo-frame
        106, # HP Printer
        112, # 32' Smart Monitor M80B UHD
        119, # Withings Body+ (Scales)
        120, # Withings Connect (Blood Pressure)
        122, # Pix-Star Easy Digital Photo Frame
        123, # HP Envy
        125  # GALAXY Watch5 Pro
    ]
}
ChatGPT_9 = {
    "SmartPlugsAndSwitches": [
        1,   # Gosuna_Socket
        6,   # Wemo_SmartPlug
        10,  # oossxx_SmartPlug
        11,  # tp-link_SmartPlug
        17,  # Lumiman_SmartPlug
        20,  # Renpho_SmartPlug
        25,  # WeMo Insight Switch
        28,  # WeMo Switch model
        34,  # TP-LinkPlugHS110
        36,  # TP-LinkPlugHS100
        39,  # Homematic pluggable switch HMIP-PS
        45,  # EdimaxPlug1101W
        47,  # EdimaxPlug2101W
        65,  # Amazon Plug
        69,  # Gosund ESP_039AAF Socket
        70,  # Gosund ESP_032979 Plug
        71,  # Gosund ESP_10098F Socket
        72,  # Gosund ESP_0C3994 Plug
        73,  # Gosund ESP_1ACEE1 Socket
        74,  # Gosund ESP_147FF9 Plug
        75,  # Gosund ESP_10ACD8 Plug
        81,  # Teckin Plug
        82,  # Yutron Plug
        94,  # Belkin Wemo switch
        95,  # TP-Link Smart plug
        96,  # iHome Plug
        115  # TOPERSUN Smart Plug
    ],
    "SmartLighting": [
        2,   # Minger_LightStrip
        3,   # Smart_LightStrip
        8,   # Lumiman_Bulb600
        12,  # Smart_Lamp
        13,  # tp-link_LightBulb
        16,  # Lumiman_Bulb900
        18,  # Gosuna_LightBulb
        23,  # Philips Hue Light Switch
        68,  # Globe Lamp ESP_B1680C
        76,  # HeimVision SmartLife Radio/Lamp
        103  # Light Bulbs LiFX Smart Bulb
    ],
    "SmartCameras": [
        4,   # itTiot_Camera
        5,   # Tenvis_Camera
        14,  # D-Link_Camera936L
        15,  # Wans_Camera
        26,  # Ednet Wireless indoor IP camera
        27,  # D-Link WiFi Day Camera
        32,  # D-Link HD IP Camera
        43,  # Edimax IC-3115W Smart HD WiFi Network Camera
        54,  # AMCREST WiFi Camera
        56,  # Arlo Q Camera
        57,  # Borun/Sichuan-AI Camera
        58,  # DCS8000LHA1 D-Link Mini Camera
        59,  # HeimVision Smart WiFi Camera
        60,  # Home Eye Camera
        61,  # Luohe Cam Dog
        62,  # Nest Indoor Camera
        63,  # Netatmo Camera
        64,  # SIMCAM 1S (AMPAKTec)
        88,  # Netatmo Welcome
        89,  # TP-Link Day Night Cloud camera
        90,  # Samsung SmartCam
        91,  # Dropcam
        92,  # Insteon Camera
        93,  # Withings Smart Baby Monitor
        107, # Nest Dropcam
        110, # Netatmo Smart Indoor Security Camera
        111, # TP-Link Tapo Pan/Tilt Wi-Fi Camera
        113, # SAMSUNG Pan/Tilt 1080P Wi-Fi Camera
        114  # EZVIZ Security Camera
    ],
    "DoorbellsAndSecuritySensors": [
        9,   # Ring_Doorbell
        21,  # Chime_Doorbell
        31,  # D-Link Door & Window sensor
        42,  # D-LinkWaterSensor
        44,  # D-Link WiFi Motion sensor
        46,  # D-Link Siren
        83,  # D-Link DCHS-161 Water Sensor
        97,  # Belkin wemo motion sensor
        98,  # NEST Protect smoke alarm
        116, # Perfk Motion Sensor
        117, # NEST Protect smoke alarm (duplicate listing)
        121, # TUYA Smartdoor Bell
        126  # Ring Video Doorbell
    ],
    "VoiceAssistantsSmartSpeakers": [
        49,  # Amazon Alexa Echo Dot
        50,  # Amazon Alexa Echo Spot
        51,  # Amazon Alexa Echo Studio
        52,  # Google Nest Mini
        53,  # Sonos One Speaker
        87,  # Amazon Echo
        104, # Triby Speaker
        108, # Amazon Echo Dot (4th Gen)
        124  # Amazon Echo Show 8
    ],
    "HubsGateways": [
        24,  # D-Link Connected Home Hub
        29,  # Ednet living starter kit power Gateway
        33,  # WeMo Link Lighting Bridge model
        40,  # MAX! Cube LAN Gateway
        41,  # Osram Lightify Gateway
        48,  # Philips Hue Bridge
        55,  # Arlo Base Station
        67,  # Eufy HomeBase
        77,  # Philips Hue Bridge (duplicate listing)
        78,  # Ring Base Station AC:1236
        86,  # Smart Things (Samsung SmartThings)
        109  # Aeotec Smart Hub
    ],
    "Others": [
        7,   # LaCrosse_AlarmClock
        19,  # Ocean_Radio
        22,  # Goumia_Coffeemaker
        30,  # Withings Wireless Scale WS-30
        35,  # Smarter iKettle 2.0 water kettle
        37,  # Fitbit Aria WiFi-enabled scale
        38,  # Smarter SmarterCoffee coffee machine
        66,  # Atomi Coffee Maker
        79,  # iRobot Roomba
        80,  # Smart Board
        84,  # LG Smart TV
        85,  # Netatmo Weather Station
        99,  # Netatmo weather station (duplicate listing)
        100, # Withings Smart scale
        101, # Blipcare Blood Pressure meter
        102, # Withings Aura smart sleep sensor
        105, # PIX-STAR Photo-frame
        106, # HP Printer
        112, # 32' Smart Monitor M80B UHD
        118, # Netatmo Weather Station (duplicate listing)
        119, # Withings Body+ (Scales)
        120, # Withings Connect (Blood Pressure)
        122, # Pix-Star Easy Digital Photo Frame
        123, # HP Envy
        125  # GALAXY Watch5 Pro
    ]
}
ChatGPT_10 =  {
  "Security": [
    4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 42, 43, 44, 46,
    54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 83, 88, 89, 90,
    91, 92, 93, 97, 98, 107, 110, 111, 113, 114, 116, 117, 121, 126
  ],
  "Plugs & Switches": [
    1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65,
    69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115
  ],
  "Lighting": [
    2, 3, 8, 12, 13, 16, 18, 23, 68, 76, 103
  ],
  "Voice Assistants & Smart Speakers": [
    49, 50, 51, 52, 53, 87, 104, 108, 124
  ],
  "Health & Fitness": [
    30, 37, 100, 101, 102, 119, 120, 125
  ],
  "Hubs & Gateways": [
    24, 29, 33, 40, 41, 48, 55, 67, 77, 78, 86, 109
  ],
  "Appliances & Others": [
    7, 19, 22, 35, 38, 66, 79, 80, 84, 85, 99, 105,
    106, 112, 118, 122, 123
  ]
}
runs = [ChatGPT_1, ChatGPT_2, ChatGPT_3, ChatGPT_4, ChatGPT_5, ChatGPT_6, ChatGPT_7, ChatGPT_8, ChatGPT_9, ChatGPT_10]
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
    "Plugs & Switches",
    "Smart Plugs & Switches",
    "Plugs/Switches",
    "Plugs and Switches",
    "Plugs_Switches",
    "SmartPlugsAndSwitches"
  ],
  2: [
    "Lighting",
    "Smart Lighting",
    "Lights",
    "Lights & Bulbs",
    "SmartLighting"
  ],
  3: [
    "Cameras",
    "Smart Cameras",
    "SmartCameras",
    "Cameras & Doorbells",
    "Security",
    'Doorbells & Security',
    "Doorbells",
  ],
  5: [
    "Voice Assistants & Smart Speakers",
    "Smart Speakers & Voice Assistants",
    "Voice Assistants",
    "Voice Assistants & Speakers",
    "VoiceAssistantsSmartSpeakers",
    'Voice_Assistants'
  ],
  4: [
    "Hubs & Gateways",
    "Hubs/Gateways",
    "Hubs / Gateways",
    "Hubs_Gateways",
    "HubsGateways",
    "Hubs/Gateways/Bridges",
    "Hubs / Gateways / Security Sensors"
  ],
  6: [
    "Sensors & Alarms",
    "Sensors",
    "Sensors, Alarms & Doorbells",
    "Sensors_Security",
    "DoorbellsAndSecuritySensors",
    
  ],
  7: [
    "Appliances & Others",
    "Others",
    "Home Appliances & Misc",
    "Smart Appliances & Others",
    "Misc / Appliances",
    "Appliances_Other",
    "Health & Fitness",
    "Health & Wearables"
  ]
}

names = {
  "Plugs & Switches": 1,
  "Lighting": 2,
  "Cameras": 3,
  "Voice Assistants & Smart Speakers": 5,
  "Security & Sensors": 6,
  "Appliances & Others": 7,
  "Hubs & Gateways": 4,
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