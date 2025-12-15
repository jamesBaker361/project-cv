import retro
import pygame
import numpy as np
import os
import time
from PIL import Image
import argparse
import csv
from shared import game_state_dict
import random



if __name__=='__main__':

    

    state_help=""
    for k,v in game_state_dict.items():
        state_help+=f"{k} has states: "+",".join(v)+"\n"

    parser=argparse.ArgumentParser()
    parser.add_argument("--game",type=str,default='SonicTheHedgehog2-Genesis',help="one of \n SuperMarioWorld-Snes \n CastlevaniaBloodlines-Genesis-v0 \n SonicTheHedgehog2-Genesis ")
    parser.add_argument("--state",type=str,default='EmeraldHillZone.Act1',help=state_help)
    parser.add_argument("--save_dir",type=str,default="videos")
    parser.add_argument("--iterations",type=int,default=1)
    parser.add_argument("--automate",action="store_true")
    parser.add_argument("--frames",type=int,default=600)

    args=parser.parse_args()

    # ---- SETTINGS ----
    save_parent=os.path.join(args.save_dir,args.game,args.state)
    os.makedirs(save_parent, exist_ok=True)
    for g in range(args.iterations):
        n_games=len(os.listdir(save_parent))
        print(f"{save_parent} has {n_games}")
        save_dir=os.path.join(save_parent , str(n_games))
        os.makedirs(save_dir,exist_ok=True)

        # Create environment with recording enabled
        env = retro.make(
            game=args.game,
            state=args.state,
            record=save_dir,
            use_restricted_actions=retro.Actions.ALL
        )

        # ---- BUTTON MAPPING ----
        # retro uses a fixed order of buttons depending on the game:
        # Example Genesis mapping: ['B','NULL','C','A','Start','Up','Down','Left','Right']
        BUTTONS = env.buttons
        print(BUTTONS)
        print(env.unwrapped.buttons)

        # Keyboard → Genesis button
        KEY_TO_BUTTON = {
            pygame.K_d: "RIGHT",
            pygame.K_a: "LEFT",
            pygame.K_w: "UP",
            pygame.K_s: "DOWN",

            pygame.K_q: "B",      # Jump
            pygame.K_e:"A",
            #pygame.K_x: "LEFT",
            #.K_c: "C",

            pygame.K_RETURN: "Start",
        }

        # Reverse lookup: button name → index
        button_index = {b: i for i, b in enumerate(BUTTONS)}

        # ---- Init pygame ----
        pygame.init()
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption(" Controller - Focus this window to play!")

        clock = pygame.time.Clock()

        obs = env.reset()
        done = False

        # Recorded list of full button vectors
        action_log = []

        print("Controls:")
        print(" AWASD = Move")
        print(" Q = Buttons (jump/spin)")
        print(" Enter = Start")
        print(" ESC = Quit")
        print("------------------")
        PATH="videos"
        os.makedirs(PATH,exist_ok=True)
        count=0
        action = np.zeros(len(BUTTONS), dtype=np.int8)
        obs, rew, terminated, truncated, info=env.step(action)
        env.reset()
        output_dict={
            key:[] for key in info
        }
        output_dict["action"]=[]
        output_dict["save_path"]=[]
        print([k for k in output_dict])
        last_button=None
        try:
            for count in range(args.frames):

                button=last_button

                if last_button in ["RIGHT","LEFT"]:
                    if random.random()<0.25:
                        if random.random() <0.5:
                            if last_button=="RIGHT":
                                button="LEFT"
                            else:
                                button="RIGHT"
                        else:
                            button="B"
                else:
                    if random.random()<0.5:
                        button="RIGHT"
                    else:
                        button="LEFT"

                action = np.zeros(len(BUTTONS), dtype=np.int8)
                idx = button_index.get(button, None)
                if idx is not None:
                    action[idx] = 1

                if np.sum(action)<1:
                    output_dict["action"].append("None")
                else:
                    output_dict["action"].append(button)
                obs, rew, terminated, truncated, info=env.step(action)

                image=obs
                image = Image.fromarray(image).resize((256,256))
                save_path_image=os.path.join(save_dir,f"{count}.jpg")
                image.save(save_path_image)
                env.render()

                # Save user action
                action_log.append(action.copy())

                clock.tick(60)  # limit to 60 FPS for smooth control

                for key in info:
                    output_dict[key].append(info[key])
                output_dict['save_path'].append(save_path_image)
                last_button=button

        except KeyboardInterrupt:
            pass

        finally:
            env.close()
            pygame.quit()

            # Save the action sequence in numpy format
            #np.save(ACTIONS_FILE, np.array(action_log, dtype=np.int8))
            #print(f"Saved {len(action_log)} actions to {ACTIONS_FILE}")

            #print("\nA .bk2 file has also been created in:", RECORD_DIR)

            with open(os.path.join(save_dir,"data.csv"), "w+") as outfile:

                # pass the csv file to csv.writer function.
                writer = csv.writer(outfile)

                # pass the dictionary keys to writerow
                # function to frame the columns of the csv file
                writer.writerow(output_dict.keys())
            
                # make use of writerows function to append
                # the remaining values to the corresponding
                # columns using zip function.
                writer.writerows(zip(*output_dict.values()))
        if args.automate==False:
            break
