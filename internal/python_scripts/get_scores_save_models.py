
import pandas as pd
import os
import numpy as np
import subprocess
import pathlib
import wandb


user_name = "venkatesh.ramesh"
name_csv="wandb_export_2023-10-10T18 45 25.429+02 00.csv"
save_dir_parent="pretrained_models"
on_remote=False #if true on local else on cluster
hydra_from_cloud=True #if true trying to restore configs from wandb, else copying from scratch folders

# wandb information
entity="causalpaca"
project="emulator"

# read in csv
df=pd.read_csv(name_csv)


# for now, we simply want to get all ckpts and run info and store them accordingly
for i,row in df.iterrows():
    name=row['Name']
    id=row['ID']
    tags=row['Tags']
    try:
        model_name=row['model/_target_'].split('.')[-1]
    
    except AttributeError:
        model_name='no_name'
        print(model_name)
 
    if "finetuning_emulator" in tags:
        parent_folder_dir="finetuning_emulator"
    elif "single_emulator" in tags:
        parent_folder_dir="single_emulator"
    elif "super_emulation" in tags:
        parent_folder_dir="super_emulator"
    else: 
        print("No experiment type found!")
        print(name, id, tags, model_name)
        exit(0)
   
    # save dir according to experiment type, ml model and experiment
    #  experiment name before optimizer
    experiment_name="".join(map(str,tags)).replace(", ","_")
    save_dir=f"{save_dir_parent}/{parent_folder_dir}/{model_name}/{experiment_name}/{id}/"
    #print(save_dir)
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    created=row['Created'].replace('-','').split('T')[0]

    if on_remote:
        checkpoint_dir_parts=f"{user_name}@mila:{row['dirs/ckpt_dir']}".split('checkpoints')
        run_dir=f"{user_name}@mila:{row['dirs/wandb_save_dir']}wandb/run-{created}_"
        checkpoint_dir=f"{checkpoint_dir_parts[0]}emulator/{id}/checkpoints/"
    else:
        checkpoint_dir_parts=f"{row['dirs/ckpt_dir']}".split('checkpoints')
        run_dir=f"{row['dirs/wandb_save_dir']}wandb/run-{created}_"
        checkpoint_dir=f"{checkpoint_dir_parts[0]}emulator/{id}/checkpoints/"
    print("checkpoint_dir", checkpoint_dir)
    print("run dir", run_dir)
    print("Copying for id:", id)

    if hydra_from_cloud:
        run_path=f"{entity}/{project}/{id}"
        run = wandb.Api(timeout=77).run(run_path)
    
        # Download from wandb cloud
        wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=save_dir)
        overrides=[]
        try:
            wandb.restore("hydra_config.yaml", **wandb_restore_kwargs)
            kwargs = dict(config_path="../../", config_name="hydra_config.yaml")
        except ValueError:  # hydra_config has not been saved to wandb :(
            print("no config found on cloud")
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
        p = subprocess.Popen(["scp", "-r", f"{run_dir}*-{id}", f"{save_dir}/"])
        sts = os.waitpid(p.pid, 0)

    #scp -r charlotte.lange@mila:/network/scratch/c/charlotte.lange/climart_out_dir/checkpoints/122ual94/* /home/charlie/Documents/uni/BA/wandb_backup/mlp_base_ensemble/6/
    p = subprocess.Popen(["scp", "-r", checkpoint_dir, save_dir])
    sts = os.waitpid(p.pid, 0)

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

