import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
import json
from transformers import AutoImageProcessor, AutoModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from torch import nn

import torch
import time
import torch.nn.functional as F
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderKL
from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.image_helpers import concat_images_horizontally
from data_loaders import SequenceGameDatasetHF
from torchvision.transforms import functional
import wandb
from classification_model import ClassificationModel,all_states,game_state_dict

try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from huggingface_hub import create_repo,HfApi,hf_hub_download
from experiment_helpers.init_helpers import default_parser,repo_api_init
from shared import SONIC_GAME,game_state_dict,all_states,all_games,NONE_STRING
import numpy as np

parser=default_parser()
parser.add_argument("--src_dataset",type=str,default="jlbaker361/CastlevaniaBloodlines-Genesis_Level1-1_10_coords")
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--use_lora",action="store_true")
parser.add_argument("--sequence_length",type=int,default=4)
parser.add_argument("--pretrained",action="store_true")
parser.add_argument("--desired_sequence_length",type=int,default=8)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--n_layers",type=int,default=4)
parser.add_argument("--unet_epochs",type=int,default=1)
parser.add_argument("--classifier_checkpoint",type=str,default="jlbaker361/ivg-class-50")
parser.add_argument("--test_only",action="store_true")

DIM_PER_TOKEN=768

all_games_plus_none=all_games+[NONE_STRING]
all_states_plus_none=all_states+[NONE_STRING]

class SequenceEncoder(torch.nn.Module):
    def __init__(self, 
                 sequence_length:int,
                 desired_sequence_length:int,
                 n_layers:int=4,
                 token_dim:int=DIM_PER_TOKEN, pretrained:bool=False,
                 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers=[]
        kernels_3d=[]
        _length=sequence_length
        desired_sequence_length=min(desired_sequence_length,sequence_length)
        while _length>desired_sequence_length:
            kernels_3d.append(True)
        print("kernels 3d",len(kernels_3d))
        self.pretrained=pretrained
        
        layers=[]
        
        if pretrained:
            for k in kernels_3d:
                layers+=[nn.Conv1d(384,384,2,2,1),nn.LeakyReLU(),nn.BatchNorm1d(384)]
            self.final_layer=torch.nn.Linear(384,token_dim)
        else:
            channels=[2**(n+2) for n in range(n_layers)]
            kernels_3d+=[False for _ in range(n_layers)]
            kernels_3d=kernels_3d[:n_layers]
        
            for chan,k in zip(channels,kernels_3d):
                if k:
                    layers+=[torch.nn.Conv3d(chan,chan*2,(4,4,4),(2,2,2),(1,1,1))]
                else:
                    layers+=[torch.nn.Conv3d(chan,chan*2,(1,4,4),(1,2,2),(0,1,1))]
                layers+=[torch.nn.LeakyReLU(),torch.nn.BatchNorm3d(chan*2)]
            self.final_layer=torch.nn.Linear(chan*2,token_dim)
            
        self.layers=torch.nn.Sequential(*layers)
        self.all_layers=torch.nn.ModuleList([self.layers,self.final_layer])
    
    def forward(self, sequence,*args, **kwds):
        if self.pretrained: 
            sequence=sequence.permute(0,2,1) #(B,N,C) -> (B,C,N)
            sequence=self.layers(sequence) # (B,C,N) -> (B,C,n) n<<<N
            sequence=sequence.permute(0,2,1) # (B,C,n) -> (B,n,C)                 
        else: 
            sequence=sequence.permute(0,2,1,3,4) #(B,N,c,H,W) -> (B,c,N,H,W)
            #print(sequence.size())
            sequence=self.layers(sequence) #  (B,c,N,H,W) -> (B,C,n,h,w)
            #print(sequence.size())
            sequence=sequence.flatten(-3,-1) # (B,C,n,h,w) -> (B,C,nhw)
            #print(sequence.size())
            sequence=sequence.permute(0,2,1) # (B,C,nhw) -> (B,nhw,C)
            #print(sequence.size())
        return self.final_layer(sequence)

def main(args):
    api,accelerator,device=repo_api_init(args)
    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)
    
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device)
    print("execution device",pipe._execution_device)
    setattr(pipe, "safety_checker",None)
    unet=pipe.unet
    accelerator.print("len params  ",len([p for p in unet.parameters()]))
    accelerator.print("len weight dict ",len(unet.state_dict()))
    vae=pipe.vae.to(device)
    vae.requires_grad_(False)
    
    image_processor=pipe.image_processor
    unet.to(device)
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(json.loads(open(hf_hub_download(
        "stabilityai/stable-diffusion-3-medium-diffusers","scheduler/scheduler_config.json")).read()))
    ddim_scheduler=DDIMScheduler()
    
    data=SequenceGameDatasetHF(
        args.src_dataset,
        image_processor,[],args.sequence_length,pretrained=args.pretrained,dim=(args.dim,args.dim)
    )
    
    n_actions=data.n_actions
    n_tokens=len(data.token_list)
    
    # Split the dataset
    train_loader,test_loader,val_loader=split_data(data,0.2,args.batch_size)
    for batch in train_loader:
        break
    if args.use_lora:
        unet.train(False)
        unet.add_adapter(
            LoraConfig(
                r=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"]
            )
        )
    action_encoder=torch.nn.Embedding(n_actions,DIM_PER_TOKEN).to(device)
    token_encoder=torch.nn.Embedding(n_tokens,DIM_PER_TOKEN).to(device)
    
    image_encoder=SequenceEncoder(args.sequence_length,args.desired_sequence_length,pretrained=args.pretrained,n_layers=args.n_layers)
    
    if args.pretrained:
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        dino_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    
    params=[p for p in action_encoder.parameters()]+[p for p in image_encoder.parameters()]
    unet_params=[p for p in unet.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,args.lr)
    unet_optimizer=torch.optim.AdamW(unet_params,args.lr)
    
    optimizer,unet,action_encoder,train_loader,test_loader,val_loader,scheduler,ddim_scheduler,image_encoder,unet_optimizer = accelerator.prepare(
        optimizer,unet,action_encoder,train_loader,
        test_loader,val_loader,scheduler,ddim_scheduler,
        image_encoder,unet_optimizer)
    
    save,load=save_and_load_functions({
        "pytorch_weights.safetensors":unet,
        "action_pytorch_weights.safetensors":action_encoder,
        "token_pytorch_weights.safetensors":token_encoder,
        "image_pytorch_weights.safetensors":image_encoder
    },save_subdir,api,args.repo_id)
    
    start_epoch=load(False)
    
    accelerator.print("starting at ",start_epoch)
    def save(*args,**kwargs):
        pass

    test_video_mse_list=[]
    game_loss_list=[]
    state_loss_list=[]
    combined_game_loss_list=[]
    combined_state_loss_list=[]
    classification_model=ClassificationModel(256,len(all_states)+1,1+len(game_state_dict))
    weight_path=hf_hub_download(args.classifier_checkpoint,"pytorch_weights.safetensors")
    if torch.cuda.is_available():
        classification_model.load_state_dict(torch.load(weight_path,weights_only=True))
    else:
        classification_model.load_state_dict(torch.load(weight_path,weights_only=True,map_location=torch.device('cpu')))

    ce_loss = nn.CrossEntropyLoss()
    
    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,misc_dict):
        sequence=batch["sequence"]
        action=batch["action"]
        image=batch["image"]
        tokens=batch["tokens"]
        mask=batch["mask"]
        bsz=image.size()[0]
        if misc_dict["mode"]!="test" and args.test_only:
            return  torch.tensor([0])
        
        if misc_dict["epochs"]>args.unet_epochs:
            unet.requires_grad_(False)
        
            if not args.pretrained:
                with torch.no_grad():
                    sequence=torch.stack([vae.encode(s).latent_dist.sample()*vae.config.scaling_factor for s in sequence])
            
            if training:
                with torch.no_grad():
                    image=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
                    
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
                    timesteps = timesteps.long()
                    
                    noise=torch.randn_like(image)
                    
                    mask=functional.resize(mask,image.size()[-2:])
                    
                    noisy_image = ddim_scheduler.add_noise(image,noise,timesteps)
                with accelerator.accumulate(params):
                    with accelerator.autocast():
                        action_embedding=action_encoder(action)
                        token_embedding=token_encoder(tokens)
                        sequence_embedding=image_encoder(sequence)
                        
                        if misc_dict["b"]==0 and misc_dict["epochs"]==start_epoch:
                            print('image.size(),action.size(),tokens.size(),sequence.size()',image.size(),action.size(),tokens.size(),sequence.size())
                            print('action_embedding.size(),token_embedding.size(),sequence_embedding.size()',action_embedding.size(),token_embedding.size(),sequence_embedding.size())
                        
                        encoder_hidden_states=torch.cat([action_embedding,token_embedding,sequence_embedding],dim=1)
                        
                        predicted=unet(noisy_image, timesteps, encoder_hidden_states, return_dict=False)[0]
                        
                        predicted=predicted*mask
                        image=image*mask
                        
                        loss=F.mse_loss(predicted.float(),image.float())
                    
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                if (misc_dict["mode"]=="val" and misc_dict["b"]==0) or (misc_dict["mode"]=="test" and misc_dict["b"]<10):
                    
                    action_embedding=action_encoder(action)
                    token_embedding=token_encoder(tokens)
                    sequence_embedding=image_encoder(sequence)
                    encoder_hidden_states=torch.cat([action_embedding,token_embedding,sequence_embedding],dim=1)
                    predicted=pipe(num_inference_steps=args.num_inference_steps
                                ,prompt_embeds=encoder_hidden_states,height=args.dim,width=args.dim,output_type="pt").images
                    loss=F.mse_loss(predicted.float(),image.float())
                    predicted_pil=image_processor.postprocess(predicted)
                    real_pil=image_processor.postprocess(image)
                    mode=misc_dict["mode"]
                    for i,(real,pred) in enumerate(zip(predicted_pil,real_pil)):
                        concat=concat_images_horizontally([real,pred])
                        accelerator.log({
                            f"{mode}_{i}":wandb.Image(concat)
                        })

                    

                    
                    if misc_dict["mode"]=="test":

                        accelerator.print("testing")

                        if args.dim!=256:
                            predicted=functional.resize(predicted,(256,256))
                        pred_state,pred_game=classification_model(predicted)

                        state_index=tokens[:,0]
                        #print(state_index[0].item(),type(state_index[0].item()),all_states_plus_none)
                        state_names=[data.token_list[i] for i in state_index]
                        new_state_index=torch.tensor([all_states_plus_none.index(sn) for sn in state_names])
                        state_one_hot=torch.stack([F.one_hot(s,len(all_states_plus_none)) for s in new_state_index]).float()

                        #print(all_games_plus_none)
                                                
                        game_index=tokens[:,1]
                        game_names=[data.token_list[i] for i in game_index]
                        new_game_index=torch.tensor([all_games_plus_none.index(gn) for gn in game_names])
                        game_one_hot=torch.stack([F.one_hot(g,len(all_games_plus_none)) for g in new_game_index]).float()

                        state_loss=ce_loss(pred_state,state_one_hot)
                        game_loss=ce_loss(pred_game,game_one_hot)

                        state_loss_list.append(state_loss.detach().cpu().numpy())
                        game_loss_list.append(game_loss.detach().cpu().numpy())

                        loss+=state_loss+game_loss
                        
                        #video testing
                        null_sequence=torch.zeros_like(sequence) #front of list is most recent
                        null_sequence_embedding=image_encoder(null_sequence)
                        #encoder_hidden_states=torch.cat([action_embedding,token_embedding,null_sequence_embedding],dim=1)
                        index=0
                        action_sequence=batch["action_sequence"]
                        #print(action_sequence.size())
                        predicted_pt_list=[]
                        predicted_pil_list=[]
                        while index<min(args.desired_sequence_length,args.sequence_length):
                            null_sequence_embedding=image_encoder(null_sequence)
                            action=action_sequence[:,index].unsqueeze(1)
                            action_embedding=action_encoder(action)
                            if index==0:
                                print(action_embedding.size(),token_embedding.size(),null_sequence_embedding.size())
                            encoder_hidden_states=torch.cat([action_embedding,token_embedding,null_sequence_embedding],dim=1)
                            predicted=pipe(num_inference_steps=args.num_inference_steps
                                ,prompt_embeds=encoder_hidden_states,height=args.dim,width=args.dim,output_type="pt").images
                            predicted_pt_list.append(predicted.detach().cpu().clone())
                            predicted_pil=image_processor.postprocess(predicted)
                            predicted_pil_list.append(predicted_pil)
                            if args.pretrained:
                                inputs = dino_processor(images=predicted_pil, return_tensors="pt")
                                outputs = dino_model(**inputs)
                                predicted=outputs.pooler_output.unsqueeze(1)
                            else:
                                predicted=torch.stack([vae.encode(p.unsqueeze(0)).latent_dist.sample()*vae.config.scaling_factor for p in predicted])
                                #print(null_sequence.size(),null_sequence_embedding.size(),predicted.size())
                            null_sequence=torch.cat([predicted,null_sequence],dim=1)
                            null_sequence=null_sequence[:,:-1,...]
                            index+=1
                        
                        print('torch.stack(predicted_pt_list).size(),batch[sequence].size()',torch.stack(predicted_pt_list).size(),batch["sequence"].size())
                        if args.pretrained:
                            video_mse=F.mse_loss(torch.stack(predicted_pt_list).permute(1,0,2,3,4),batch["image_sequence"])
                        else:
                            video_mse=F.mse_loss(torch.stack(predicted_pt_list).permute(1,0,2,3,4),batch["sequence"])
                        
                        test_video_mse_list.append(video_mse.detach().cpu().numpy())

                            
                        
                        loss=video_mse


                else:
                    loss=torch.tensor([0])
        else:
            encoder_hidden_states=torch.zeros((bsz,1,DIM_PER_TOKEN),device=device)
            
            if training:
                
                image=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()
                
                noise=torch.randn_like(image)
                noisy_image = ddim_scheduler.add_noise(image,noise,timesteps)
                with accelerator.accumulate(unet_params):
                    with accelerator.autocast():
                        
                        if misc_dict["b"]==0 and misc_dict["epochs"]==start_epoch:
                            print('image.size()',image.size())
                            
                        predicted=unet(noisy_image, timesteps,encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                        
                        loss=F.mse_loss(predicted.float(),image.float())
                    
                    accelerator.backward(loss)
                    unet_optimizer.step()
                    unet_optimizer.zero_grad()
            elif misc_dict["b"]==0:
                predicted=pipe(prompt_embeds=encoder_hidden_states,
                               num_inference_steps=args.num_inference_steps,height=args.dim,width=args.dim,output_type="pt").images
                loss=F.mse_loss(predicted.float(),image.float())
                mode=misc_dict["mode"]
                for i,gen_image in enumerate(predicted):
                    accelerator.log({
                        f"unet_{mode}_{i}":wandb.Image(gen_image)
                    }) 
            else:
                loss=torch.tensor([0])
            
        
        return loss.cpu().detach().item()
        '''
        {"sequence":sequence,
               "action":F.one_hot(torch.tensor(
                   self.action_list.index( row["action"])),self.n_actions),
               "mask":mask,
               "tokens":tokens,
               "score":row["template_score"],
               "image":self.image_processor.preprocess(row["image"])[0]
               }'''
    batch_function()

    for batch in train_loader:
        break

    for game in  all_games_plus_none:
        for state in all_states_plus_none:
            g=data.token_list.index(game)
            s=data.token_list.index(state)
            action_embedding=action_encoder(torch.tensor([[0]]))
            token_embedding=token_encoder(torch.tensor([[g,  s]]))
            null_sequence=torch.zeros_like(batch["sequence"][0].unsqueeze(0))
            if not args.pretrained:
                with torch.no_grad():
                    null_sequence=torch.stack([vae.encode(s).latent_dist.sample()*vae.config.scaling_factor for s in null_sequence])
            null_sequence_embedding=image_encoder(null_sequence)
            encoder_hidden_states=encoder_hidden_states=torch.cat([action_embedding,token_embedding,null_sequence_embedding],dim=1)
            predicted=pipe(num_inference_steps=args.num_inference_steps
                                ,prompt_embeds=encoder_hidden_states,height=args.dim,width=args.dim,output_type="pt").images
            new_s=all_states_plus_none.index(state)
            new_g=all_games_plus_none.index(game)
            state_one_hot=F.one_hot(torch.tensor(new_s),len(all_states_plus_none)).float().unsqueeze(0)
            game_one_hot=F.one_hot(torch.tensor(new_g),len(all_games_plus_none)).float().unsqueeze(0)

            if args.dim!=256:
                predicted=functional.resize(predicted,(256,256))
            pred_state,pred_game=classification_model(predicted)
            state_loss=ce_loss(pred_state,state_one_hot)
            game_loss=ce_loss(pred_game,game_one_hot)

            combined_state_loss_list.append(state_loss.detach().cpu().numpy())
            combined_game_loss_list.append(game_loss.detach().cpu().numpy())

            image=image_processor.postprocess(predicted)[0]

            accelerator.log({
                f"combined_{game}_{state}":wandb.Image(image)
            })



            




    '''print(test_video_mse_list)
    print(game_loss_list)
    print(state_loss_list)'''
    for name, metric_list in zip(["test video","game loss", "state loss", "combined game loss", "combined state loss"],
                                 [test_video_mse_list,game_loss_list, state_loss_list, combined_game_loss_list,combined_state_loss_list]):
        accelerator.print(f"\t {name} {np.mean(metric_list)}")
        print(f"\t {name} {np.mean(metric_list)}")
    #evaluate 
    
        
        
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