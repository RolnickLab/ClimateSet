
import pandas as pd
import os
import wandb
import subprocess
import pathlib

"""
Creating a backup of models from a csv with best checkpiont per unuique tag.
"""

missing_ids=[]

name_csv="/home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/emulator/internal/scores/best_per_tag_checkpoints.csv"
save_dir_parent="pretrained_models"
on_remote=False #if true on local else on cluster
user_name="charlotte.lange" # for if copying from cluster

# for meta files
files_parent='/home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/runs/wandb'
list_foldernames_runs=os.listdir(files_parent)

# wandb information
entity="causalpaca"
project="emulator"

# read in csv
df=pd.read_csv(name_csv)


# for now, we simply want to get all ckpts and run info and store them accordingly
for i,row in df.iterrows():


    name=row['name']
    id=row['id']
    tags=row['tags']
    checkpoint_path=row['checkpoint_path']
   
    if "finetuning_emulator" in tags:
        parent_folder_dir="finetuning_emulator"
    elif "single_emulator" in tags:
        parent_folder_dir="single_emulator"
    elif "super_emulation" in tags:
        parent_folder_dir="super_emulator"
    else: 
        print("No experiment type found!")
        print(name, id, tags)
        exit(0)

    # get ml model name
    if "unet" in tags:
        model_name="Unet"
    elif "conv_lstm" in tags:
        model_name="CNN_LSTM_ClimateBench"
    elif "climax_frozen" in tags:
        model_name="ClimaX_frozen"
    elif "climax" in tags:
        model_name="ClimaX"
    else:
        print("Cannot infer model from tag.")
        print(tags)
        exit(0)
   
    # save dir according to experiment type, ml model and experiment
    #  experiment name before optimizer
    experiment_name="".join(map(str,tags)).replace(", ","_").replace("'","").replace('[',"").replace(']','')
    print("experiment_name", experiment_name)
    save_dir=f"{save_dir_parent}/{parent_folder_dir}/{model_name}/{experiment_name}/{id}"
    #print(save_dir)
    print(save_dir)
    
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
   
    print("id")
    # get dir for configs
    config_dir=[s for s in list_foldernames_runs if s.endswith(id)]
    print(config_dir)
   
    if len(config_dir)>1:
        print("no unique run specification, options:")
        print(config_dir)
    elif len(config_dir)==0:
        print("no config dir found!")
        print(id)
        

        # Try to recover from cloud
        print('trying to recover from cloud')
        if True:
            run_path=f"{entity}/{project}/{id}"
            run = wandb.Api(timeout=77).run(run_path)
        
            # Download from wandb cloud
            wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=f"{save_dir}/files")
            overrides=[]
            try:
                wandb.restore("hydra_config.yaml", **wandb_restore_kwargs)
                wandb.restore("config.yaml", **wandb_restore_kwargs)
                wandb.restore("output.log", **wandb_restore_kwargs)
                wandb.restore("requirements.txt", **wandb_restore_kwargs)
                wandb.restore("wandb-metadata.json", **wandb_restore_kwargs)
                wandb.restore("wandb-summary.json", **wandb_restore_kwargs)
                #kwargs = dict(config_path="../../", config_name="hydra_config.yaml")
            except ValueError:  # hydra_config has not been saved to wandb :(
                print("no config found on cloud")
                missing_ids.append(id)
                overrides += json.load(
                    wandb.restore("wandb-metadata.json", **wandb_restore_kwargs)
                )["args"]
                kwargs = dict()
                if len(overrides) == 0:
                    raise ValueError(
                        "wandb-metadata.json had no args, are you sure this is correct?"
                    )
                    # also wandb-metadata.json is unexpected (was likely overwritten)
            
            print("recovered config")
    else:
        config_dir=config_dir[0]

        # get folder name
        config_dir=os.path.join(files_parent, config_dir)
        config_dir=os.path.join(config_dir, 'files')
        
        
        if on_remote:
            config_dir=f"{user_name}@mila:{config_dir}"
        # save configs
        print("copying meta files")
        print(os.listdir(config_dir))
        p = subprocess.Popen(["scp", "-r", f"{config_dir}/", f"{save_dir}/"])
        sts = os.waitpid(p.pid, 0)

    # save checkpoint
    print("copying checkpoint")
    if on_remote:
        checkpoint_path=f"{user_name}@mila:{checkpoint_path}"
    
    #scp -r charlotte.lange@mila:/network/scratch/c/charlotte.lange/climart_out_dir/checkpoints/122ual94/* /home/charlie/Documents/uni/BA/wandb_backup/mlp_base_ensemble/6/
    p = subprocess.Popen(["scp", "-r", checkpoint_path, f"{save_dir}/checkpoints/"])
    sts = os.waitpid(p.pid, 0)

  

print("FINISHED")
print("missing ids")
print(missing_ids)
"""

reduce="sum"
metrics_list=['test/rsuc/rmse', 'test/rsuc/rmse']

if reduce=='sum':
    df['score'] = df[metrics_list].sum(axis=1)
elif reduce=='mean':
    df['score'] = df[metrics_list].mean(axis=1)
else:
    print("Uknown reduce")
    raise NotImplementedError

df=df.sort_values('score', ascending=False)
print(df)
df.to_csv(name_csv)

best_model=df['ID'][0]

print(f"Best performing model of group {group_name} is {best_model}")

for i,row in df.iterrows():
    print(row)
    # figure out how to do scp from python script
    name=row['Name']
    id=row['ID']
    checkpoint_dir=f"charlotte.lange@mila:{row['model_checkpoint/dirpath']}/*"
    save_dir=group_name+"/"+name+'_'+id+"/"
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)

    created=row['Created'].replace('-','').split('T')[0]
    run_dir=f"charlotte.lange@mila:{row['dirs/wandb_save_dir']}wandb/run-{created}_"
    #print("run dir", run_dir)
    print("Copying for id:", id)
    p = subprocess.Popen(["scp", "-r", f"{run_dir}*-{id}", f"{save_dir}/"])
    sts = os.waitpid(p.pid, 0)
    #scp -r charlotte.lange@mila:/network/scratch/c/charlotte.lange/climart_out_dir/checkpoints/122ual94/* /home/charlie/Documents/uni/BA/wandb_backup/mlp_base_ensemble/6/
    p = subprocess.Popen(["scp", checkpoint_dir, save_dir])
    sts = os.waitpid(p.pid, 0)

"""

