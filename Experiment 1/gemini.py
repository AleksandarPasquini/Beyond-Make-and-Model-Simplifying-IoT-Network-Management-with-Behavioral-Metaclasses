import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

Gemini1 = {
    "Smart Plugs/Outlets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lights/Bulbs/Switches": [2, 3, 8, 13, 16, 18, 23, 33, 41, 68, 76, 77, 103],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 42, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 116,117, 121, 126, 97],
    "Smart Home Hubs/Bridges/Gateways": [24, 29, 40, 48, 67, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Home Automation": [7, 12, 19, 22, 30, 35, 37, 38, 79, 80, 84, 93, 98, 99, 100, 101, 102, 119, 120, 66],
    "Smart Devices/Other": [85, 105, 106, 112, 122, 123, 125, 118, 83]
}
missing1 = [66, 83, 97]
duplicate1 = [48]
Gemini2 =  {
    "Smart Lighting": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Plugs/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 121, 126],
    "Smart Home Hubs/Bridges": [24, 29, 33, 40, 41, 48, 67, 77, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Other": [7, 12, 19, 22, 30, 35, 37, 38, 66, 79, 80, 83, 84, 85, 93, 99, 100, 101, 105, 106, 112, 118, 119, 120, 122, 123, 125],
    "Smart Sensors": [31, 42, 44, 97, 98, 102, 116, 117]
}
duplicate2 = [76, 31, 42, 44, 98, 99, 100, 101, 102, 116, 117, 118, 119, 120]
Gemini3 = {
    "Smart Lighting": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Plugs/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 116, 121, 126, 97],
    "Smart Home Hubs/Gateways": [24, 29, 33, 40, 41, 48, 67, 78, 86, 109, 77],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances": [22, 35, 38, 66],
    "Smart Sensors/Other Devices": [7, 12, 19, 30, 37, 79, 80, 84, 85, 93, 98, 99, 100, 101, 102, 105, 106, 112, 117, 118, 119, 120, 122, 123, 125, 42, 83]
}
failed3 = {
    "Smart Lighting": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Plugs/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 116, 121, 126],
    "Smart Home Hubs/Gateways": [24, 29, 33, 40, 41, 48, 67, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Other": [7, 12, 19, 22, 30, 35, 37, 38, 79, 80, 84, 85, 93, 98, 99, 100, 101, 102, 105, 106, 112, 117, 118, 119, 120, 122, 123, 125, 42, 83],
}
missing3 = [77, 97]
Gemini4 = {
    "Smart Plugs/Sockets/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Lighting": [2, 3, 8, 13, 16, 18, 23, 33, 41, 68, 76, 103],
    "Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 42, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 116, 117, 121, 126, 97],
    "Home Hubs/Gateways": [24, 29, 40, 48, 67, 78, 109, 77],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Home Appliances": [7, 19, 22, 30, 35, 37, 38, 79, 80, 84, 85, 93, 98, 99, 100, 101, 102, 118, 119, 120],
    "Other Smart Devices/Computers/Wearables": [83, 86, 105, 106, 112, 122, 123, 125, 12, 66]
}
missing4 = [12, 66, 77, 97]
Gemini5 = {
    "Smart Plugs/Sockets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lighting": [2, 3, 8, 13, 16, 18, 23, 33, 68, 76, 103],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 110, 111, 113, 114, 121, 126],
    "Smart Home Hubs/Gateways": [24, 29, 40, 41, 48, 67, 77, 78, 109],
    "Smart Speakers/Displays/Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Other": [7, 12, 19, 22, 30, 35, 37, 38, 79, 80, 84, 85, 86, 93, 99, 100, 101, 105, 106, 107, 112, 118, 119, 120, 122, 123, 125, 66],
    "Smart Sensors": [31, 42, 44, 83, 97, 98, 102, 116, 117]
}
duplicates5 = [31, 42, 44, 83, 97, 98, 99, 102, 116, 117, 118, 119, 120]
missing5 = [66]
Gemini6 = {
    "Smart Plugs/Sockets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lighting": [2, 3, 8, 13, 16, 18, 23, 33, 68, 76, 103],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 98, 107, 110, 111, 113, 114, 121, 126],
    "Smart Home Hubs/Gateways": [24, 29, 40, 41, 48, 67, 77, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Other Devices": [7, 12, 19, 22, 30, 35, 37, 38, 79, 80, 83, 84, 85, 93, 99, 100, 101, 102, 105, 106, 112, 118, 119, 120, 122, 123, 125, 66],
    "Smart Sensors": [31, 42, 44, 97, 116, 117]
}
duplicates6 = [31, 42, 44, 97, 116, 117, 118, 120]
missing6 = [66]
Gemini7 = {
    "Smart Plugs/Sockets": [1, 6, 10, 11, 17, 20, 25, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 95, 96, 115, 28, 94],
    "Smart Lights/Bulbs/Strips": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Cameras/Video": [4, 5, 14, 15, 26, 27, 32, 43, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 107, 110, 111, 113, 114, 9, 93],
    "Smart Home Hubs/Bridges/Gateways": [24, 29, 33, 40, 41, 48, 55, 67, 77, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Sensors/Detectors": [31, 42, 44, 46, 97, 98, 99, 116, 117, 118],
    "Smart Appliances/Other Devices": [7, 12, 19, 21, 22, 30, 35, 37, 38, 79, 80, 83, 84, 85, 100, 101, 102, 105, 106, 112, 119, 120, 121, 122, 123, 125, 126, 66]
}
missing7 =[9, 28, 66, 93, 94]
Gemini8 = {
    "Smart Plugs/Sockets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lights/Bulbs": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 93, 97, 98, 110, 111, 113, 114, 116, 117, 121, 126],
    "Smart Hubs/Bridges/Gateways": [24, 29, 33, 40, 41, 48, 67, 77, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Home Appliances/Sensors": [7, 12, 19, 22, 30, 35, 37, 38, 42, 79, 80, 83, 84, 85, 99, 100, 101, 102, 118, 119, 120, 125, 66],
    "Smart Electronics/Other": [105, 106, 107, 112, 122, 123]
}
missing8 = [66]
Gemini9 = {
    "Smart Plugs/Outlets": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Smart Lights/Bulbs/Strips": [2, 3, 8, 13, 16, 18, 23, 68, 76, 103],
    "Smart Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 32, 43, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 83, 89, 90, 91, 92, 107, 110, 111, 113, 114, 121, 126, 88],
    "Smart Home Hubs/Gateways": [24, 29, 33, 40, 41, 48, 67, 77, 78, 86, 109],
    "Smart Speakers/Voice Assistants": [49, 50, 51, 52, 53, 87, 104, 108, 124],
    "Smart Appliances/Home Automation": [7, 12, 19, 22, 30, 35, 37, 38, 79, 80, 84, 85, 93, 99, 100, 101, 102, 112, 118, 119, 120, 122, 123, 125, 66, 105, 106],
    "Smart Sensors": [31, 42, 44, 97, 98, 116, 117]
}
duplicates9 = [31, 44, 97, 98, 99, 116, 117, 118]
missing9 = [66, 88, 105, 106]
Gemini10 = {
    "Smart Plugs/Switches": [1, 6, 10, 11, 17, 20, 25, 28, 34, 36, 39, 45, 47, 65, 69, 70, 71, 72, 73, 74, 75, 81, 82, 94, 95, 96, 115],
    "Lighting": [2, 3, 8, 13, 16, 18, 23, 33, 68, 76, 103, 12],
    "Cameras/Security": [4, 5, 9, 14, 15, 21, 26, 27, 31, 32, 43, 44, 46, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 88, 89, 90, 91, 92, 97, 98, 110, 111, 113, 114, 116, 117, 121, 126, 107],
    "Home Hubs/Gateways & Speakers": [24, 29, 40, 41, 48, 67, 77, 78, 49, 50, 51, 52, 53, 87, 104, 108, 109, 124],
    "Home Appliances": [22, 35, 38, 66, 79, 80, 84, 106, 123, 19],
    "Health/Wellness/Sensors": [30, 37, 93, 99, 100, 101, 102, 119, 120, 125, 85, 118, 7, 42, 83],
    "Other/Smart Home Integration": [86, 105, 112, 122]
}


runs = [Gemini1, Gemini2, Gemini3, Gemini4, Gemini5, Gemini6, Gemini7, Gemini8, Gemini9, Gemini10]
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
        'Smart Plugs/Outlets', 'Smart Plugs/Switches', 'Smart Plugs/Sockets/Switches', 'Smart Plugs/Sockets', 'Smart Plugs/Sockets', 'Smart Plugs/Sockets', 'Smart Plugs/Outlets', 'Smart Plugs/Switches'
    ],
    2: [
        'Smart Lights/Bulbs/Switches', 'Smart Lighting', 'Smart Lighting', 'Lighting', 'Smart Lighting', 'Smart Lighting', 'Smart Lights/Bulbs/Strips', 'Smart Lights/Bulbs', 'Smart Lights/Bulbs/Strips', 'Lighting'
    ],
    3: [
        'Smart Cameras/Security', 'Smart Cameras/Security', 'Smart Cameras/Security', 'Cameras/Security', 'Smart Cameras/Security', 'Smart Cameras/Security', 'Smart Cameras/Video', 'Smart Cameras/Security', 'Smart Cameras/Security', 'Cameras/Security'
    ],
    4: [
        'Smart Home Hubs/Bridges/Gateways', 'Smart Home Hubs/Bridges', 'Smart Home Hubs/Gateways', 'Home Hubs/Gateways', 'Smart Home Hubs/Gateways', 'Smart Home Hubs/Gateways', 'Smart Home Hubs/Bridges/Gateways', 'Smart Hubs/Bridges/Gateways', 'Smart Home Hubs/Gateways', 'Home Hubs/Gateways & Speakers'
    ],
    5: [
        'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Displays/Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Smart Speakers/Voice Assistants', 'Home Hubs/Gateways & Speakers'
    ],
    6: [
        'Smart Appliances/Home Automation', 'Smart Appliances/Other', 'Smart Appliances', 'Home Appliances', 'Smart Appliances/Other', 'Smart Appliances/Other Devices', 'Smart Appliances/Other Devices', 'Smart Home Appliances/Sensors', 'Smart Appliances/Home Automation', 'Home Appliances',  'Other Smart Devices/Computers/Wearables', 'Other/Smart Home Integration', 'Smart Electronics/Other'
    ],
    7: [
        'Smart Devices/Other', 'Smart Sensors', 'Smart Sensors/Other Devices', 'Smart Sensors', 'Smart Sensors', 'Smart Sensors/Detectors', 'Smart Sensors', 'Health/Wellness/Sensors'
    ]
}

names = {
    "Smart Plugs/Switches/Outlets/Sockets": 1,
    "Smart Lighting/Bulbs/Strips/Switches": 2,
    "Smart Cameras/Security/Video": 3,
    "Smart Home Hubs/Bridges/Gateways": 4,
    "Smart Speakers/Voice Assistants/Displays": 5,
    "Smart Appliances/Home Automation/Other Devices": 6, 
    "Smart Sensors/Detectors/Other": 7,
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