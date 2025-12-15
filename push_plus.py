from datasets import Dataset
import csv
import os
from shared import game_state_dict,game_key_dict
import cv2 as cv
from PIL import Image
import numpy as np
from huggingface_hub import HfApi

api=HfApi()

os.makedirs("testing_save",exist_ok=True)

def mask_black_with_neighbors(img, thresh=10, min_neighbors=5):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    black = (gray < thresh).astype(np.uint8)

    # Kernel to count 8 neighbors (center excluded)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    neighbor_count = cv.filter2D(black, -1, kernel)

    # Pixel is selected if black AND neighbor_count >= min_neighbors
    mask = (black & (neighbor_count >= min_neighbors)).astype(np.uint8)

    return 1-mask

CUTOFF=0.01

black_list_dict={
    'SonicTheHedgehog2-Genesis':[f"{s}.jpg" for s in range(61,67)]+[f"{s}.jpg" for s in range(158,163)]+
    ["133.jpg","134.jpg","176.jpg","13.jpg"]+[f"{s}.jpg" for s in range(117,129)]+[f"{s}.jpg" for s in range(100,106)]+
    [f"{s}.jpg" for s in range(45,48)], #['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':[], #["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis': ["107.jpg"],
}

interval=15

button_list=["B","LEFT","RIGHT","DOWN","UP"]
for game,states_list in game_state_dict.items():


    key_list=game_key_dict[game]
    output_dict={
        key:[] for key in key_list+["image","state","game","episode","coords","template_score","step"] #LESS THEN .4 IS USUALLY SAFE
    }
    path_list=[f for f in os.listdir(os.path.join("distorted_sprite_from_sheet",f"{game}")) 
                     if f.endswith("jpg") and f not in black_list_dict[game] ]
    template_list=[
        cv.cvtColor(cv.imread( os.path.join("distorted_sprite_from_sheet",f"{game}", file) ,cv.IMREAD_COLOR),cv.COLOR_BGR2RGB) for file in 
        path_list  
    ]
    print(black_list_dict[game] )

    mask_list=[
        mask_black_with_neighbors(template) for template in template_list
    ]
    limit=500
    '''
    template_list=[
        cv.cvtColor(cv.imread( os.path.join("sprite_from_sheet",f"{game}", f"{button}.jpg") ,cv.IMREAD_COLOR),cv.COLOR_BGR2RGB) for button in button_list
    ]
    mask_list=[
        mask_black_with_neighbors(template) for template in template_list
    ]'''
    #print(len(template_list))
    for state in states_list[::-1]:
        
        directory=os.path.join("videos",str(interval),game,state)
        repo=f"jlbaker361/{game}_{state}_{interval}_{limit}_coords"
        if os.path.exists(directory):
            if api.repo_exists(repo):
                print(repo,"exists")
                continue
            else:
                count=0
                for episode in os.listdir(directory):

                    
                    #index=0
                    subdir=os.path.join(directory,episode)
                    csv_path=os.path.join(subdir,'data.csv')
                    with open(csv_path,"r") as file:
                        reader=csv.DictReader(file)
                        
                        for row in reader:
                            if count>=limit:
                                Dataset.from_dict(output_dict).push_to_hub(repo)
                                print(f"pushed {repo}")
                                break
                            output_dict["step"].append(count)
                            count+=1
                            for key,value in row.items():
                                output_dict[key].append(value)
                            cv_image=cv.cvtColor(cv.imread(row["save_path"],cv.IMREAD_COLOR),cv.COLOR_BGR2RGB)
                            output_dict["state"].append(state)
                            output_dict["game"].append(game)
                            output_dict["image"].append(Image.fromarray(cv_image))
                            output_dict["episode"].append(episode)
                            lowest=1.0
                            best_template=None
                            res_list=[]
                            for n,(template,mask,path) in enumerate(zip(template_list,mask_list,path_list)):
                                res = cv.matchTemplate(cv_image,template,cv.TM_SQDIFF_NORMED) #,mask=mask)
                                min_val, _, min_loc, max_loc = cv.minMaxLoc(res)
                                res_list.append([min_val,min_loc,max_loc,template,n,path])

                            res_list.sort(key=lambda x: x[0])
                            best=res_list[0]
                            min_val,min_loc,max_loc,best_template,best_n,path=best
                            h,w=best_template.shape[:2]
                            top_left = min_loc

                            bottom_right = (top_left[0] + w, top_left[1] + h)
    
                            '''cv.rectangle(cv_image,top_left, bottom_right, 255, 2)

                            cv_image[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=best_template

                            cv.imwrite(f"test_{count}.jpg",cv.cvtColor(cv_image,cv.COLOR_RGB2BGR))
                            print(f"test_{count}.jpg",best_n, min_val,path)'''

                            output_dict["coords"].append([top_left, bottom_right])
                            output_dict["template_score"].append(min_val)

                            #index+=1
                            
                            '''if count>10:
                                Dataset.from_dict(output_dict).push_to_hub(f"jlbaker361/{game}_{state}_10_coords")
                                print("pushed")
                                break'''
                Dataset.from_dict(output_dict).push_to_hub(repo)
                print(f"pushed {repo}")
                            