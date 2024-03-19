import os
import subprocess

# first zip on desired level then upload respective file

folder_name="pretrained_models"

#for exp in os.listdir(folder_name):
if True:
    exp="single_emulator"
    print("working on exp", exp)
    parent_folder=os.path.join(folder_name,exp)
    for m in os.listdir(parent_folder):

        print("zipping model", m)
        path=os.path.join(os.path.join(parent_folder, m))
        print("zipping" ,path)
        # zip
        #p = subprocess.Popen(["scp", "-r", checkpoint_path, f"{save_dir}/checkpoints/"])
        p = subprocess.Popen(["zip", "-r", f"{path}.zip", path])
        sts = os.waitpid(p.pid, 0)
        print("uploading")
        print(f"{path}.zip")
        # upload to arbutus
        p = subprocess.Popen(["aws", "s3", "cp", f"{path}.zip", f"s3://causalpaca/{path}.zip"])
        sts = os.waitpid(p.pid, 0)
