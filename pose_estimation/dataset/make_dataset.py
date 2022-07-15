import json
import matplotlib.pyplot as plt


with open("./annotation/person_keypoints_train2017.json", "r") as jf:
    train_json = json.load(jf)


num_keypoints = []
# print(train_json.keys())
# print(train_json['annotations'])
# print(train_json['annotations'][1000]['keypoints'])
for annotation in train_json['annotations']:
    num = annotation['num_keypoints']
    num_keypoints.append(num)

plt.hist(num_keypoints)
plt.show()