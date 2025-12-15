import datasets
from shared import game_state_dict
from datasets import load_dataset,Dataset
import sys
import numpy as np

print(sys.argv)
args=sys.argv[1:]

interval=int(args[0])
limit=int(args[1])


merged_dict={
    "game":[],
    "state":[],
    "image":[],
    "coords":[],
    "template_score":[],
    "episode":[],
    "action":[],
   # "mask":[],
    "step":[]
}
for game,state_list in game_state_dict.items():
    for state in state_list:
        path=f"jlbaker361/{game}_{state}_{interval}_{limit}_coords"
        data=load_dataset(path,split="train")
        for row in data:
            for key in merged_dict:
                merged_dict[key].append(row[key])
                
Dataset.from_dict(merged_dict).push_to_hub(f"jlbaker361/merged_ivg_{interval}_{limit}")

print("donesies :)")