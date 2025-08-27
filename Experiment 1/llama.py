import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


llama1 = {
    "Lighting": [1, 2, 3, 8, 12, 13, 18, 23, 68, 103, 16],
    "Cameras and Security": [4, 5, 10, 14, 15, 26, 27, 32, 43, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 88, 89, 90, 91, 92, 93, 110, 111, 113, 114, 121, 126, 9, 21, 31, 78, 98, 117],
    "Plugs and Sockets": [6, 17, 20, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115, 69, 77, 80],
    "Home Automation Hubs": [24, 28, 33, 40, 41, 46, 48, 67, 109, 39, 86],
    "Audio and Speakers": [49, 50, 51, 52, 104, 125, 87],
    "Sensors and Monitors": [11, 19, 29, 30, 42, 44, 45, 76, 85, 97, 101, 102, 116, 118, 119, 83, 100, 120],
    "Home Appliances": [84, 112, 124, 22, 34, 35, 36, 37, 38, 66, 94, 99, 105, 106, 107, 108, 122, 123, 25, 47, 7]
}
miss1= [7, 9, 16, 21, 22, 25, 31, 34, 35, 36, 37, 38, 39, 47, 66, 69, 77, 78, 80, 83, 86, 87, 94, 98, 99, 100, 105, 106, 107, 108, 117, 120, 122, 123]
duplicate1 = [68]
llama2 = {
    "Lighting": [2, 3, 8, 12, 13, 18, 23, 41, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 89, 90, 91, 92, 93, 108, 110, 111, 112, 113, 114],
    "Smart Plugs and Outlets": [1, 6, 10, 11, 17, 20, 28, 34, 35, 44, 45, 46, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Security and Alarm Systems": [7, 9, 16, 21, 24, 30, 31, 33, 36, 37, 38, 39, 40, 42, 48, 78, 83, 85, 97, 98, 99, 116, 117, 118, 121, 126],
    "Home Appliances": [22, 25, 55, 67, 80, 94, 105, 106, 107, 29],
    "Speakers and Audio Devices": [19, 49, 50, 51, 52, 53, 76, 104, 124],
    "Health and Wellness": [100, 101, 102, 119, 120, 109, 122, 123, 125, 84, 86, 87, 88]
}
miss2 = [7, 12, 16, 19, 29, 67, 76, 84, 86, 87, 88, 89, 109, 122, 123, 124, 125]
duplicate2 = [104]
llama3 = {
    "Lighting": [1, 2, 8, 12, 13, 18, 23, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 113, 114, 89],
    "Smart Plugs": [6, 10, 11, 17, 20, 34, 35, 36, 45, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Security and Alarm Systems": [7, 9, 16, 21, 24, 28, 30, 31, 33, 37, 38, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 67, 76, 83, 84, 85, 86, 87, 88, 97, 98, 99, 104, 105, 106, 107, 108, 109, 117, 118, 121, 122, 123, 124, 125, 126],
    "Home Appliances": [22, 25, 29, 55, 80, 94, 116],
    "Audio and Entertainment": [19, 52, 53, 78, 112],
    "Health and Wellness": [3, 100, 101, 102, 119, 120]
}
duplicates3 = [52, 53, 100, 101, 102, 119, 120]
missing3 = [89, 94, 116]
llama4 = {
    "Lighting": [2, 3, 8, 13, 18, 23, 41, 48, 68, 77, 103],
    "Security Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 112, 113, 114],
    "Smart Plugs": [1, 6, 10, 11, 17, 20, 34, 35, 44, 45, 46, 47, 66, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Home Appliances": [22, 30, 36, 37, 38, 39, 42, 67, 69, 76, 84, 101, 102, 105, 106, 107],
    "Doorbells and Sensors": [9, 16, 21, 28, 29, 31, 33, 40, 49, 50, 51, 52, 53, 78, 85, 86, 87, 88, 94, 97, 98, 99, 100, 104, 108, 109, 121, 122, 123, 124, 126],
    "Audio and Visual": [19, 25, 125],
    "Other": [7, 12, 24, 80, 83, 89, 116, 117, 118, 119, 120]
}
miss4= [121]
failed5 = {
    "Lighting": [2, 8, 13, 18, 23, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 108, 110, 111, 112, 113, 114],
    "Smart Plugs": [1, 6, 10, 11, 16, 17, 20, 28, 29, 33, 34, 35, 36, 44, 45, 46, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Security": [9, 21, 24, 30, 31, 37, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 53, 55, 83, 84, 85, 86, 87, 88, 94, 97, 98, 99, 100, 101, 102, 109, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
    "Home Appliances": [22, 25, 67, 76, 80, 104, 105, 106, 107],
    "Sensors": [3, 7, 12, 19, 77, 78],
    "Entertainment": [52, 53, 89],
    "Health and Fitness": [100, 101, 102, 119, 120]
}
llama5 = {
    "Lighting and Plugs": [1, 2, 6, 8, 10, 11, 13, 16, 17, 18, 20, 23, 28, 34, 35, 36, 44, 45, 46, 47, 66, 68, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 103, 115, 76],
    "Security and Surveillance": [4, 5, 9, 14, 21, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 113, 114, 121, 126, 89],
    "Home Automation": [7, 15, 24, 25, 29, 30, 31, 40, 41, 42, 48, 67, 77, 85, 98, 109, 55, 78],
    "Audio and Entertainment": [19, 49, 50, 51, 52, 53, 104, 105, 124],
    "Appliances": [22, 33, 37, 38, 39, 80, 100, 106, 107, 108, 112, 122, 123, 125],
    "Sensors and Monitoring": [101, 102, 116, 117, 118, 119, 120, 83, 97, 99],
    "Other": [3, 12, 84, 86, 87, 88, 94]
}
miss5= [55, 76, 78, 83, 89, 94, 97, 99]
failed6 = {
    "Lighting": [1, 2, 8, 13, 18, 23, 68, 103],
    "Security and Surveillance": [4, 5, 9, 14, 21, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 113, 114, 121, 126],
    "Smart Plugs and Outlets": [6, 10, 11, 16, 17, 20, 28, 34, 35, 36, 44, 45, 46, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Home Automation Hubs and Gateways": [24, 29, 30, 40, 41, 48, 67, 77, 109],
    "Sensors and Monitoring": [7, 15, 25, 31, 42, 44, 45, 85, 98, 101, 102, 116, 117, 118, 119, 120],
    "Audio and Entertainment": [19, 49, 50, 51, 52, 53, 104, 105, 124],
    "Appliances and Gadgets": [22, 33, 37, 38, 39, 80, 100, 106, 107, 108, 112, 122, 123, 125],
    "Other": [3, 12, 84, 86, 87, 88]
}
llama6 = {
    "Lighting": [2, 3, 8, 13, 18, 23, 34, 68, 103, 12],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 79, 89, 90, 91, 92, 93, 108, 110, 111, 113, 114],
    "Smart Plugs": [1, 6, 10, 11, 17, 20, 35, 36, 44, 45, 46, 47, 65, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Home Security": [9, 16, 21, 25, 28, 48, 66, 67, 69, 77, 78, 85, 86, 87, 88, 98, 99, 100, 101, 102, 117, 118, 119, 120, 121, 122, 123, 124, 126],
    "Speakers": [49, 50, 51, 52, 53, 104, 125, 7, 19, 76],
    "Sensors": [31, 42, 97, 116, 83],
    "Appliances": [22, 33, 37, 38, 39, 40, 41, 80, 105, 106, 107, 112, 84, 24, 29, 30, 55, 94, 109]
}
duplicates6 = [49, 50, 51, 52, 53, 125, 31, 42, 97, 116, 32, 68, 112, 84]
missing6 = [7, 12, 19, 24, 29, 30, 55, 76, 83, 94, 109]
llama7 = {
    "Smart Plugs and Sockets": [
        1, 6, 10, 11, 17, 20, 35, 36, 45, 47, 65, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115, 94
    ],
    "Lighting": [
        2, 3, 8, 12, 13, 18, 23, 33, 43, 68, 76, 103
    ],
    "Security Cameras": [
        4, 5, 14, 15, 26, 27, 32, 41, 44, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 79, 89, 90, 91, 92, 110, 111, 112, 114, 9, 21, 107
    ],
    "Home Appliances": [
        22, 34, 37, 38, 39, 42, 46, 49, 50, 51, 66, 67, 77, 84, 98, 78, 105, 106, 109
    ],
    "Sensors and Alarms": [
        7, 16, 19, 24, 25, 28, 29, 30, 31, 40, 48, 53, 69, 85, 86, 87, 88, 97, 99, 100, 104, 116, 117, 118, 83
    ],
    "Audio and Visual": [
        52, 80, 93, 113, 108,
    ],
    "Health and Wellness": [
        101, 102, 119, 120, 121, 122, 123, 124, 125, 126
    ]
}
missing7= [9, 21, 78, 83, 94, 105, 106, 107, 108, 109]
failed7 = {
    "Lighting": [2, 3, 8, 12, 13, 18, 23, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 33, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 108, 110, 111, 112, 113, 114],
    "Smart Plugs": [1, 6, 10, 11, 17, 20, 34, 35, 44, 45, 46, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Home Security": [9, 16, 21, 25, 28, 30, 31, 36, 37, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 53, 67, 76, 78, 84, 85, 86, 87, 88, 94, 97, 98, 99, 100, 101, 102, 109, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
    "Home Appliances": [22, 24, 29, 55, 80, 104, 105, 106, 107],
    "Audio Devices": [49, 50, 51, 52, 53, 77, 104],
    "Sensors": [31, 42, 44, 97, 116],
    "Wearables/Health": [100, 101, 102, 119, 120, 125]
}
llama8 = {
    "Lighting": [2, 3, 8, 12, 13, 18, 23, 68, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 33, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 108, 110, 111, 112, 113, 114, 89],
    "Smart Plugs": [1, 6, 10, 11, 17, 20, 34, 35, 45, 46, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Home Security": [9, 16, 21, 25, 28, 30, 36, 37, 38, 39, 40, 41, 48, 67, 76, 78, 84, 85, 86, 87, 88, 94, 98, 99, 109, 117, 118, 121, 122, 123, 124, 126],
    "Home Appliances": [22, 24, 29, 55, 80, 105, 106, 107, 7],
    "Audio Devices": [49, 50, 51, 52, 53, 77, 104, 19],
    "Sensors and Wearables": [31, 42, 44, 97, 100, 101, 102, 116, 119, 120, 125, 83]
}
missing8 = [7, 19, 83, 89]
duplicates8 = [49, 50, 51, 52, 53, 104, 31, 42, 44, 97, 100, 101, 102, 116, 119, 120, 125]
llama9 = {
    "Smart Lighting": [2, 3, 8, 12, 13, 18, 23, 68, 104],
    "Security and Surveillance": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 110, 111, 113, 114, 121, 126, 89, 93],
    "Smart Plugs and Outlets": [1, 6, 10, 17, 20, 29, 34, 35, 36, 45, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115, 11, 94],
    "Smart Home and Automation": [7, 16, 19, 22, 24, 25, 28, 33, 38, 39, 40, 41, 46, 48, 67, 76, 80, 84, 85, 86, 87, 88, 103, 105, 106, 107, 108, 109, 118, 122, 123, 124, 125, 55, 77, 99],
    "Audio and Entertainment": [49, 50, 51, 52, 53, 112],
    "Health and Wellness": [30, 37, 78, 100, 101, 102, 119, 120],
    "Sensors": [31, 42, 44, 97, 98, 116, 117, 83]
}
duplicates9 = [10, 49, 50, 51, 52, 53, 105, 22, 30, 37, 38, 66, 80, 101, 102, 106, 107, 119, 120, 31, 42, 44, 116, 117]
missing9 = [11, 55, 77, 83, 89, 93, 94, 99, 112]
failed9 = {
    "Smart Lighting": [2, 3, 8, 12, 13, 18, 23, 68, 77, 104],
    "Security Cameras": [4, 5, 10, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 110, 111, 112, 113, 114],
    "Smart Plugs": [1, 6, 10, 17, 20, 29, 34, 35, 36, 45, 47, 66, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Smart Home Devices": [7, 9, 16, 19, 21, 22, 24, 25, 28, 30, 31, 33, 37, 38, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 52, 53, 67, 76, 80, 84, 85, 86, 87, 88, 101, 102, 103, 105, 106, 107, 108, 109, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
    "Sensors": [31, 42, 44, 97, 98, 116, 117],
    "Audio Devices": [49, 50, 51, 52, 53, 104, 105],
    "Appliances": [22, 38, 66, 78, 80, 106, 107],
    "Doorbells": [9, 21, 121, 126]
}
llama10 = {
    "Lighting": [1, 2, 3, 8, 12, 13, 18, 23, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 112, 113, 114, 88, 89],
    "Smart Plugs": [6, 10, 11, 16, 17, 20, 28, 34, 35, 36, 45, 47, 66, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115, 69],
    "Security": [9, 21, 25, 31, 42, 44, 46, 78, 83, 97, 98, 116, 117, 121, 126],
    "Home Appliances": [22, 33, 38, 39, 40, 41, 48, 55, 67, 85, 94, 104, 106, 107, 7, 24, 29, 80, 84, 86, 99, 118, 122, 123, 125],
    "Audio and Speakers": [49, 50, 51, 52, 53, 105, 108, 109, 124, 19, 76, 87],
    "Health and Wellness": [30, 37, 100, 101, 102, 119, 120]
}
missing10 =  [7, 19, 24, 29, 69, 76, 80, 84, 86, 87, 88, 89, 99, 118, 122, 123, 125]
duplicate10 = [37, 100]
failed10 = {
    "Lighting": [2, 8, 13, 18, 23, 68, 77, 103],
    "Cameras": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 79, 90, 91, 92, 93, 110, 111, 112, 113, 114],
    "Smart Plugs": [1, 6, 10, 11, 16, 17, 20, 28, 29, 30, 34, 35, 36, 45, 47, 66, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115],
    "Security": [9, 21, 25, 31, 42, 44, 46, 49, 50, 51, 52, 53, 67, 69, 76, 78, 80, 83, 84, 85, 86, 87, 88, 97, 98, 99, 101, 102, 105, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
    "Home Appliances": [22, 33, 37, 38, 39, 40, 41, 48, 55, 89, 94, 100, 104, 106, 107, 108, 109],
    "Audio": [24, 48, 49, 50, 51, 52, 53, 77, 105],
    "Sensors": [3, 7, 12, 19, 31, 42, 44, 97, 98, 102, 116, 117],
    "Health and Wellness": [30, 37, 100, 101, 102, 119, 120]
}
concepts = {
    1: ['Lighting', 'Lighting', 'Lighting', 'Lighting', 'Lighting and Plugs', 'Lighting', 'Lighting', 'Smart Lighting', 'Lighting', 'Lighting'],
    2: ['Cameras and Security', 'Security and Alarm Systems', 'Security and Alarm Systems', 'Security Cameras', 'Security and Surveillance', 'Home Security', 'Home Security', 'Security Cameras', 'Security and Surveillance', 'Security', 'Cameras'],
    3: ['Plugs and Sockets', 'Smart Plugs and Outlets', 'Smart Plugs', 'Smart Plugs', 'Smart Plugs and Sockets', 'Smart Plugs', 'Smart Plugs', 'Smart Plugs and Outlets', 'Smart Plugs'],
    4: ['Home Automation Hubs', 'Smart Home and Automation'],
    5: ['Audio and Speakers', 'Speakers and Audio Devices', 'Audio and Entertainment', 'Audio and Visual', 'Audio and Entertainment', 'Speakers', 'Audio Devices', 'Audio and Entertainment', 'Audio and Speakers'],
    6: ['Sensors and Monitors', 'Health and Wellness', 'Health and Wellness', 'Doorbells and Sensors', 'Sensors and Monitoring', 'Sensors', 'Sensors and Wearables', 'Sensors', 'Health and Wellness', 'Sensors and Alarms'],
    7: ['Other', 'Other', 'Appliances', 'Home Appliances', 'Home Appliances', 'Home Appliances', 'Home Automation', 'Home Appliances', 'Home Appliances',]
}

names = {
    'Lighting': 1,
    'Security and Surveillance': 2,
    'Power and Plugs': 3,
    'Home Automation': 4,
    'Audio and Visual': 5,
    'Sensors and Health': 6,
    'Other':7
}
runs = [llama1, llama2, llama3, llama4, llama5, llama6, llama7, llama8, llama9, llama10]
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