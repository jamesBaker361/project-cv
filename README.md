## Step One
Install the requirements file. Skip to step 5 if you don't want to generate your own data (stable retro doesn't work on all systems for some reason. It DOES work on the UMBC Chip cluster though)

## Step Two
Download the ROMs for Sonic, Castlvania and Mario [try this](https://www.romsgames.net/roms/sega-genesis/) and/or [this](https://www.romsgames.net/roms/super-nintendo/)  and save them to a folder called zipfiles

## Step Three
run 'rom_import.py'

## Step Four
You can now generate your own data using 'play_automated.py'! 
Use 'push_plus.py' to upload each game dataset individually, and then use 'combine_datasets.py' to merge them all

## Step Five
Train 'classification_model.py' or skip this if you want to use a pretrained one

## Step Six
Run 'inversion_ivg.py' If you don't have your own data, use it with arg "--src_dataset  jlbaker361/merged_ivg_15_500"  and if you didn't train your own classifier use it with arg "--classifier_checkpoint jlbaker361/ivg-class-50 "# project-cv
