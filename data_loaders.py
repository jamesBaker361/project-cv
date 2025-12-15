from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import AutoencoderKL
from datasets import load_dataset
import datasets
import torch
from PIL import Image
from shared import SONIC_GAME,game_state_dict,all_states,all_games,NONE_STRING
import random
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import resize

import torch.nn.functional as F

NULL_ACTION=35 #this is the "button" pressed for null frames ()


def find_earliest_less_than(arr, target):
    left, right = 0, len(arr) - 1
    result = None

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < target:
            result = arr[mid]      # candidate found
            right = mid - 1        # but try to find earlier one
        else:
            left = mid + 1

    return result

class ClassificationDataset(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor):
        super().__init__()
        self.data = load_dataset(src_dataset, split="train")

        try:
            self.data = self.data.cast_column("image", datasets.Image())
        except Exception as e:
            print("map error ",e)

        self.image_processor=image_processor
        self.all_games=all_games+[NONE_STRING]
        self.all_states=all_states+[NONE_STRING]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess(self.data[index]["image"])[0],
            "game":F.one_hot(torch.tensor(self.all_games.index(self.data[index]["game"])),len(self.all_games)).float(),
            "state":F.one_hot(torch.tensor(self.all_states.index(self.data[index]["state"])),len(self.all_states)).float(),
        }


    

class SequenceGameDatasetHF(Dataset):
    def __init__(self, src_dataset, image_processor, metadata_key_list=[], 
                 sequence_length:int=2,
                 process=False,
                 dim=(256,256),
                 vae=None,
                 threshold=0.4,
                 pretrained=False
                 ):
        super().__init__()
        self.data = load_dataset(src_dataset, split="train")

        try:
            self.data = self.data.cast_column("image", datasets.Image())
        except Exception as e:
            print("map error ",e)

        self.image_processor = image_processor
        self.metadata_key_list = metadata_key_list
        self.vae = vae
        self.threshold=threshold
        self.pretrained=pretrained
        
        if self.pretrained:
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
            self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")


        self.n_actions = len(set(self.data["action"]))
        self.action_list=list(set(self.data["action"]))

        self.index_list = []
        self.seqence_length=sequence_length
        self.dim=dim
        self.token_list=list(set([t for t in self.data["game"]]+[t for t in self.data["state"]]))+[NONE_STRING]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        sequence=[]
        action_sequence=[]
        image_sequence=[]
        game_index=self.token_list.index(row["game"])
        state_index=self.token_list.index(row["state"])
        none_index=self.token_list.index(NONE_STRING)
        for j in range(1,self.seqence_length+1):
            if i-j <0:
                img=Image.new('RGB',self.dim,'black')
                past_action="B"
            elif self.data[i-j]["episode"]!=self.data[i]["episode"]:
                img=Image.new('RGB',self.dim,'black')
                past_action=self.data[i-j]["action"]
            else:
                img=self.data[i]["image"]
                past_action=self.data[i-j]["action"]
            img=img.resize(self.dim)
            image_sequence.append(img)
            if self.pretrained:
                inputs = self.processor(images=img, return_tensors="pt")
                outputs = self.model(**inputs)
                img=outputs.pooler_output[0]
            sequence.append(img)
            action_sequence.append(past_action)
        tokens=[]       
        if  self.data[i]["template_score"]<self.threshold:
            coords=self.data[i]["coords"]
            (x1,y1),(x2,y2)=coords
            if random.random()<0.5:
                tokens+=[none_index,game_index]
                mask=torch.zeros(self.dim,dtype=torch.bool)
                mask[y1:y2,x1:x2]=True
            else:
                tokens+=[state_index,game_index]
                mask=torch.ones(self.dim,dtype=torch.bool)
                mask[y1:y2,x1:x2]=False
        else:
            mask=torch.ones(self.dim,dtype=torch.bool)
            tokens+=[state_index,game_index]
            
        mask=mask.unsqueeze(0).expand([4,-1,-1])
        tokens=torch.tensor(tokens)
        action=torch.tensor([self.action_list.index( row["action"])])
        try:
            action_sequence=torch.tensor([self.action_list.index(p_action.upper()) for p_action in action_sequence])
        except ValueError:
            action_sequence=torch.tensor([self.action_list.index(p_action) for p_action in action_sequence])
                    
        if not self.pretrained:
            sequence=self.image_processor.preprocess(sequence)

        else:
            sequence=torch.stack(sequence)
        out = {"sequence":sequence,
               "action":action,
               "mask":mask,
               "tokens":tokens,
               "action_sequence":action_sequence,
               #"score":row["template_score"],
               "image":self.image_processor.preprocess(row["image"].resize(self.dim))[0],
               "image_sequence":self.image_processor.preprocess(image_sequence)
               }
        return out
        

class ImageDatasetHF(Dataset):
    def __init__(self,src_dataset:str,
                 image_processor:VaeImageProcessor,
                 skip_num:int=1):
        super().__init__()
        dataset=load_dataset(src_dataset,split="train")
        
        dataset=dataset.cast_column("image",datasets.Image())
        data=dataset["image"]
        self.image_processor=image_processor
        if image_processor is not None:
            _image_list=[self.image_processor.preprocess(image)[0] for image in data]
        else:
            _image_list=data
        self.image_list=_image_list[::skip_num]
        

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image=torch.tensor(self.image_list[index])
        
        return {
            "image":image
        }
    
class RenderingModelDatasetHF(Dataset):
    def __init__(self, src_dataset, image_processor, metadata_key_list=[], process=False, vae=None):
        super().__init__()
        self.data = load_dataset(src_dataset, split="train")

        try:
            self.data = self.data.cast_column("image", datasets.Image())
        except Exception as e:
            print("map error ",e)

        self.image_processor = image_processor
        self.metadata_key_list = metadata_key_list
        self.vae = vae

        # preprocess metadata if needed
        if process:
            self.n_actions = len(set(self.data["action"]))
            '''if image_processor is not None:
                self.data = self.data.map(
                    lambda x: {"image": image_processor.preprocess(x["image"])[0]},
                    batched=False
                )'''
            '''self.data = self.data.map(
                lambda x: {"action": F.one_hot(torch.tensor(x["action"]), self.n_actions)},
                batched=False
            )'''
        else:
            self.n_actions = len(self.data["action"][0])

        # -------------------------------------------- #
        #         BUILD ONLY INDEX PAIRS (cheap)        #
        # -------------------------------------------- #

        self.index_list = []

        episodes = self.data["episode"]
        N = len(self.data)

        for i in range(1,N ):
            # skip crossing episodes
            if episodes[i] != episodes[i - 1]:
                continue
            self.index_list.append(i)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        i = self.index_list[idx]

        row = self.data[i]
        past_row = self.data[i - 1]

        img = row["image"]
        past_img = past_row["image"]

        if self.image_processor:
            img = self.image_processor.preprocess(img)[0]
            past_img = self.image_processor.preprocess(past_img)[0]

        out = {"image": img, "past_image": past_img}
        
        # metadata
        for k in self.metadata_key_list: # ["action"]:
            out[k] = torch.tensor(row[k])
            
        out["action"]=F.one_hot(torch.tensor(row["action"]),self.n_actions)
        return out

    
    
class VelocityPositionDatasetHF(Dataset):
    def __init__(self,src_dataset:str,image_processor:VaeImageProcessor=None,process:bool=False):
        super().__init__()
        self.data=load_dataset(src_dataset,split="train")
        try:
            self.data=self.data.cast_column("image",datasets.Image())
        except:
            pass
        
        self.start_index_list=[]
        episode_set=set()
        self.initial_velocity_x=[]
        self.initial_velocity_y=[]
        
        if process:
            self.n_actions=len(set(self.data["action"]))
            if image_processor is not None:
                self.data=self.data.map(lambda x :{"image": image_processor.preprocess( x["image"])[0]})
            self.data=self.data.map(lambda x: {"action":F.one_hot(torch.tensor(x["action"]),self.n_actions)})
        else:
            self.n_actions=len(self.data["action"][0])
        
        for i,row in enumerate(self.data):
            if row["episode"] not in episode_set:
                episode_set.add(row["episode"])
                self.start_index_list.append(i)
                self.initial_velocity_x.append(0)
                self.initial_velocity_y.append(0)
            else:
                prior_row=self.data[i-1]
                d_x=row["x"]-prior_row["x"]
                d_y=row["y"]-prior_row["y"]
                self.initial_velocity_x.append(d_x)
                self.initial_velocity_y.append(d_y)
        self.start_index_list.append(i)
        
    def __len__(self):
        return len(self.data)-1
    
    def __getitem__(self, index)->dict:
        output= {
            k:torch.tensor(self.data[k][index]) for k in ["action","image","x","y"]
        }
        output["vi_x"]=self.initial_velocity_x[index]
        output["vi_y"]=self.initial_velocity_y[index]
        
        output["xf"]=self.data["x"][index+1]
        output["yf"]=self.data["y"][index+1]
        
        output["vf_x"]=self.initial_velocity_x[index+1]
        output["vf_y"]=self.initial_velocity_y[index+1]
        
        return output

if __name__=="__main__":
    image_processor=VaeImageProcessor()
    data=SequenceGameDatasetHF("jlbaker361/CastlevaniaBloodlines-Genesis_Level1-1_10_coords",image_processor,process=True,pretrained=True)
    data=torch.utils.data.DataLoader(data,batch_size=2,shuffle=True)
    for row in data:
        print("?")
        break
    sequence=row["sequence"].squeeze(1)
    print(sequence.size())
    data=SequenceGameDatasetHF("jlbaker361/CastlevaniaBloodlines-Genesis_Level1-1_10_coords",image_processor,process=True,)
    data=torch.utils.data.DataLoader(data,batch_size=2,shuffle=True)
    for row in data:
        print("?")
        break
    sequence=row["sequence"].squeeze(1)
    print(sequence.size())
    processed_image=image_processor.postprocess(sequence)
    print(type(processed_image),type(processed_image[0]),len(processed_image))
    processed_image[0].save("first.jpg")
    mask=row["mask"]
    print(mask.size())
    t=sequence*mask
    print(t.size())
    #image_processor.postprocess(t)[0].save("second.jpg")
    for key,value in row.items():
        print(key,type(value),value.dtype)