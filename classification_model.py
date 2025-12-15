import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
import json
from constants import VAE_WEIGHTS_NAME
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from torch import nn

import torch
import time
import torch.nn.functional as F
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderKL
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.image_helpers import concat_images_horizontally
from data_loaders import ClassificationDataset,NONE_STRING
from shared import game_state_dict,all_states
from diffusers.image_processor import VaeImageProcessor

try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from huggingface_hub import create_repo,HfApi,hf_hub_download
from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()
parser.add_argument("--src_dataset",type=str,default="jlbaker361/classification-ivg-reps-3")
parser.add_argument("--dim",type=int,default=256)

class ClassificationModel(torch.nn.Module):
    def __init__(self, dim:int,n_state:int,n_sprite:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
        self.n_state=n_state
        self.n_sprite=n_sprite
        chan=4
        zero=torch.zeros((1,3,dim,dim))
        self.layers=[torch.nn.Conv2d(3,chan,1,1),torch.nn.BatchNorm2d(chan),torch.nn.LeakyReLU()]
        while dim > 4:
            self.layers+=[torch.nn.Conv2d(chan,2*chan,4,2,1),torch.nn.BatchNorm2d(chan*2),torch.nn.LeakyReLU()]
            dim=dim //2
            chan*=2
        self.layers+=[torch.nn.Conv2d(chan,2*chan,2,2),torch.nn.BatchNorm2d(chan*2),torch.nn.LeakyReLU()]
        self.layers+=[nn.Flatten()]
        for l in self.layers:
            #print(zero.size())
            zero=l(zero)
        features=zero.size()[-1]
        #print(features)

        self.layers+=[nn.Linear(features,512),nn.LeakyReLU(),nn.Dropout(0.1),nn.Linear(512,256),nn.LeakyReLU(),nn.Dropout(0.1)]

        self.state_layers=torch.nn.Sequential(*[nn.Linear(256,64),nn.LeakyReLU(),nn.Dropout(0.1),nn.Linear(64,n_state)])
        self.sprite_layers=torch.nn.Sequential(*[nn.Linear(256,64),nn.LeakyReLU(),nn.Dropout(0.1),nn.Linear(64,n_sprite)])

        self.module_list=torch.nn.ModuleList(self.layers+[self.sprite_layers,self.state_layers])

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.state_layers(x),self.sprite_layers(x)




def main(args):
    api,accelerator,device=repo_api_init(args)
    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)
    model=ClassificationModel(args.dim,len(all_states)+1,1+len(game_state_dict)) #adding 1 for NONE string
    model=model.to(device)

    dataset=ClassificationDataset(args.src_dataset,VaeImageProcessor())

    # Split the dataset
    train_loader,test_loader,val_loader=split_data(dataset,0.8,args.batch_size)
    for batch in train_loader:
        break

    params=[p for p in model.parameters()]
    optimizer=torch.optim.AdamW(params,args.lr)

    optimizer,train_loader,test_loader,val_loader,model=accelerator.prepare(optimizer,train_loader,test_loader,val_loader,model)

    save,load=save_and_load_functions({
        "pytorch_weights.safetensors":model
    },save_subdir,api,args.repo_id)
    
    start_epoch=load(False)
    
    accelerator.print("starting at ",start_epoch)

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,misc_dict):
        ce_loss = nn.CrossEntropyLoss()
        image=batch["image"]
        state=batch["state"]
        game=batch["game"]

        if misc_dict["epochs"]==start_epoch and misc_dict["b"]==0:
            accelerator.print('image.size(),state.size(),game.size()',image.size(),state.size(),game.size())

        if training:
            with accelerator.accumulate(params):
                with accelerator.autocast():
                    pred_state,pred_game=model(image)
                    state_loss=ce_loss(pred_state,state)
                    game_loss=ce_loss(pred_game,game)
                    loss=game_loss+state_loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        else:
            pred_state,pred_game=model(image)
            state_loss=ce_loss(pred_state,state)
            game_loss=ce_loss(pred_game,game)
            loss=game_loss+state_loss

        accelerator.log({
            "state":state_loss.cpu().detach().numpy(),
            "game":game_loss.cpu().detach().numpy()
        })
        return loss.cpu().detach().numpy()
    
    batch_function()



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")