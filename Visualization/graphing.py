import matplotlib.pyplot as plt
import numpy as np 
from csv import DictReader
from csv import reader

# This file is for presenting 
# a visualization of the distrubution of all of the classes in the LISA dataset

# Making a labels map (dictionary - keys, values) of our dataset
labels_map = {
    0: "addedLane",
    1: "curveLeft",
    2: "curveRight",
    3: "dip",
    4: "doNotEnter",
    5: "doNotPass",
    6: "intersection",
    7: "keepRight",
    8: "laneEnds",
    9: "merge",
    10: "noLeftTurn",
    11: "noRightTurn",
    12: "pedestrianCrossing",
    13: "rampSpeedAdvisory20",
    14: "rampSpeedAdvisory35",
    15: "rampSpeedAdvisory40",
    16: "rampSpeedAdvisory45",
    17: "rampSpeedAdvisory50",
    18: "rampSpeedAdvisoryUrdbl",
    19: "rightLaneMustTurn",
    20: "roundabout",
    21: "school",
    22: "schoolSpeedLimit25",
    23: "signalAhead",
    24: "slow",
    25: "speedLimit15",
    26: "speedLimit25",
    27: "speedLimit30",
    28: "speedLimit35",
    29: "speedLimit40",
    30: "speedLimit45",
    31: "speedLimit50",
    32: "speedLimit55",
    33: "speedLimit65",
    34: "speedLimitUrdbl",
    35: "stop",
    36: "stopAhead",
    37: "thruMergeLeft",
    38: "thruMergeRight",
    39: "thruTrafficMergeLeft",
    40: "truckSpeedLimit55",
    41: "turnLeft",
    42: "turnRight",
    43: "yield",
    44: "yieldAhead",
    45: "zoneAhead25",
    46: "zoneAhead45",
}

all_file = "./LISA_DATA/allAnnotations.csv" 
num_classes = 47
count = 0 
occurrence_list = [0] * num_classes
    
with open(all_file, 'r') as read_obj:
    csv_reader = DictReader(read_obj)
    for row in csv_reader:
        class_value = int(row['ClassID'])
        # Grab the current count of the class
        count = occurrence_list[class_value]
        # Increment by 1 
        count += 1
        # Update the count of the class 
        occurrence_list[class_value] = count

# For the 47 classes
a = np.arange(47)
fig, ax = plt.subplots(figsize=(12,10), edgecolor='k')
ax.set_xticks(a)
# Getting the labels from the 'labels_map' dictionary
values = list(labels_map.values())
ax.set_xticklabels(values, rotation=90)
ax.set_title('LISA Dataset Class Distribution')

plt.xlabel('Class')
plt.ylabel('Count')
ax.bar(values, occurrence_list)
plt.show()