import os
import cv2 as cv
import numpy as np
from shared import crop_black,crop_black_gray,game_state_dict
from PIL import Image

def is_almost_white(img, thresh=250, min_fraction=0.95):
    """
    img: uint8 grayscale or RGB
    thresh: pixel is considered white if >= thresh
    min_fraction: minimum percentage of white pixels
    """
    if img.ndim == 3:  # RGB → convert to grayscale intensity
        gray = img.mean(axis=2)
    else:
        gray = img

    white_mask = gray >= thresh
    fraction = white_mask.mean()   # between 0 and 1

    return fraction >= min_fraction


for game in game_state_dict:
    os.makedirs(os.path.join("sprite_from_sheet", game), exist_ok=True)

    spritesheet=cv.imread(f"spritesheets/{game}.png",cv.IMREAD_GRAYSCALE)
    spritesheet_rgb=cv.imread(f"spritesheets/{game}.png",cv.IMREAD_COLOR_RGB)

    print(np.mean(spritesheet))

    binary=(spritesheet!=255).astype(np.uint8)*255

    Image.fromarray(binary).save(f"binary_{game}.jpg")

    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

    areas = stats[:, cv.CC_STAT_AREA]

    n=1
    for label in range(1,n_labels):
        component_mask = (labels == label).astype(np.uint8)
        con_components = cv.bitwise_and(spritesheet_rgb, spritesheet_rgb, mask=component_mask)
        if is_almost_white(con_components)==False:
            #component_list.append(con_components)

            if areas[label]>200:
                
                #continue
                '''else:
                print(n,areas[label])'''
            
                # Save component for inspection
                try:
                    con_components=crop_black(con_components)
                    Image.fromarray(con_components).save(

                        os.path.join("sprite_from_sheet", game, f"{n}.jpg")
                    )
                    n+=1
                except ValueError:
                    pass