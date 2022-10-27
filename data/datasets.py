import torch
import numpy as np
import os
from mother_data.utils.constants import VAR_SOURCE_LOOKUP
from data_paths import PROCESSED_DATA
from mother_data.utils.helper_funcs import get_keys_from_value
import h5py
from torch.utils.data import  DataLoader
import random
from typing import Dict, Optional, List, Callable, Tuple, Union
import shutil
import zipfile
from data_transform import AbstractTransform 
from data_normalization import Normalizer
from data_normalization import NormalizationMethod
import pickle

class Causalpaca_HdF5_Dataset(torch.utils.data.Dataset): # TODO: where to do the training vs testing discrimination? 
    """Class for working with multiple HDF5 datasets"""

    def __init__(
            self,
            years: List[int],
            experiments: List[str],
            variables: List[str],
            name: str = "test",
            data_dir: str = None,
            load_h5_into_mem: bool = False,
            models : List[str] = ["NorESM2-LM"],
            ensemble_members: List[str] = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
            freq : str = "mon",
            forcing_nom_res: str = "50_km",
            model_nom_res: str= "250_km",
            tau : int = 4,
            instantaneous:bool = False,
            target: str = 'emulator' # other choice is 'causal' leading to a different choice of base dataset / sampling
        ):

        self._name=name
        self.years=years
        self.n_years =len(years)
        self.experiments=experiments
        self.variables=variables
        self.models=models
        self.ensemble_members=ensemble_members
        self.freq = freq
        self.forcing_nom_res=forcing_nom_res
        self.model_nom_res=model_nom_res
        self.tau=tau
        self.instantaneous=instantaneous

        # if no data dir is given, take default 
        if data_dir is None:
            self.data_dir=PROCESSED_DATA
        else:
            self.data_dir=data_dir

        self.load_h5_into_mem=load_h5_into_mem
     
        have_forcing=False
        have_model=False   
        self.model_variables = []
        self.forcing_variables = []

        for v in variables:

            t = get_keys_from_value(VAR_SOURCE_LOOKUP, v)
            if t == "model":
                self.model_variables.append(v)
                have_model=True
            elif t == "raw":
                self.forcing_variables.append(v)
                have_forcing=True

            else:
                print(
                    f"WARNING: unknown source type for var {v}. Not supported. Skipping."
                )

        if not(have_forcing):
            print("WARNING: No forcing variables were given, please complete your specifications.")
            exit(0)
        if not(have_model):
            print("WARNING: No forcing variables were given, please complete your specifications.")
            exit(0)
      

        # shared dset kwargs
        dset_kwargs = dict(
            data_dir=self.data_dir,
            years=self.years,
            freq=self.freq,
            tau=self.tau,
            instantaneous=self.instantaneous
        ) 

    

        if self.load_h5_into_mem:
            print("WARNING: Load into mem not yet implemented")
            if target=='emulator':

                # given emulator, we get batch size many experiment_variables pairs sample batch_size examples form them
                # each sample contains all possible year
                dset_class = Causalpaca_HdF5_FastSingleDataset_Emulator
            elif target=='causal':
                raise NotImplementedError
                dset_class_f = Causalpaca_HdF5_FastSingleDataset_Causal

        else:
            if target=='emulator':
                dset_class = Causalpaca_HdF5_SingleDataset_Emulator
            elif target=='causal':
                print("WARNING: Causal Dataloader not yet implemented")
                raise NotImplementedError
                dset_class = Causalpaca_HdF5_SingleDataset_Causal

        dataset_size_forcing = 0 # TODO: check up if forcing data n matches model n 
        dataset_size_model = 0
        self.h5_dsets_forcing= {} # create list of dset objects
        self.h5_dsets_model = {}

        if have_forcing:
            # onne forcing dataset per experiment
            for exp in self.experiments: 

                try:
                    exp_h5_dset_forcing=dset_class(experiment='ssp119', variables=self.forcing_variables, nom_res=forcing_nom_res, aspect='forcing', **dset_kwargs) # redo Debugging
                except ValueError:
                    continue

                n_samples =len(exp_h5_dset_forcing)
                # TODO: read out some stuff? 
                self.h5_dsets_forcing[exp]=exp_h5_dset_forcing
                dataset_size_forcing+= n_samples
                
                # copy to slurm
                exp_h5_dset_forcing.copy_to_slurm_tmp_dir(aspect='forcing')

            self.dataset_size_forcing=dataset_size_forcing

        if have_model:
            for exp in self.experiments:
               
                self.h5_dsets_model[exp]=[]
                for  model in self.models:
                    for member in self.ensemble_members:
                        
                        try:
                            exp_h5_dset_model=dset_class(model=model, experiment='ssp126', member=member, variables=self.model_variables, nom_res=model_nom_res, aspect='model', **dset_kwargs)
                        except ValueError:
                            continue
                        n_samples =len(exp_h5_dset_model)
                         
                        if n_samples==0:
                            print(f"wARNING: No data available for pairing: model {model} member {member} exp {exp}. Skipping.")
                            continue
                        
                        dataset_size_model+= n_samples
                       
                        self.h5_dsets_model[exp].append(exp_h5_dset_model)

                        # copy to slurm
                        exp_h5_dset_model.copy_to_slurm_tmp_dir(aspect='model')
               
            """
            # TODO: test if it works
            for exp in self.experiments:
                
                for dset in self.h5_dsets_forcing[exp]:
                    dset.copy_to_slurm_tmp_dir(aspect='forcing')
                for dset in self.h5_dsets_model[exp]:
                    dset.copy_to_slurm_tmp_dir(aspect='model')
            """
    @property
    def name(self):
        return self.name.upper()

    def __str__(self):
        s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
        return s

    def __len__(self):
        return len(self.years)

    
    def __getitem__(self, item): ## add type signature

        # choose a random experiment 
        exp=random.choice(self.experiments)
        # load forcing for a random experiment, load all years
        forcing=self.h5_dsets_forcing[exp]
        raw_Ys = forcing["Data"]

        # choose data belonging to a random model / member pair
        model_data=random.choice(self.h5_dsets_model[exp])
        raw_Xs = model_data["Data"]

        if self.load_h5_into_mem:
            X = raw_Xs["Data"]
            Y = raw_Ys["Data"]
        
        else:
            # possabilty to apply transforms and normalizations 
            # TODO
            X=raw_Xs["Data"]
            Y=raw_Ys["Data"]
     
        return X, Y

    def output_dim(self):
        return self.h5_dsets_forcing[0].spatial_dim 
        
    def input_dim(self) -> Dict[str, int]:
        return self.h5_dsets_model[0].spatial_dim

    def close(self):
        for dset in [self.h5_dsets_model+self.h5_dsets_forcing]:
            dset.close()
        

class Causalpaca_HdF5_SingleDataset_Emulator(torch.utils.data.Dataset):

    def __init__(self,
            experiment : str,
            variables : List[str],
            data_dir : str,
            years : List[int],
            freq : str,
            nom_res : str,
            aspect : str, #either 'model' or 'forcing'
            model: Optional[str] = "",
            member: Optional[str] = "",
            tau : int = 0, # not relevant for emulator part
            instantaneous : bool = False, # not relevant for emulator part
            
            ):
      
        self.experiment=experiment
        self.data_dir=data_dir
        self.variables=variables
        self.years=years
        
        self.freq=freq
        self.nom_res=nom_res
        self.aspect=aspect
        self.tau=tau
        self.instantaneous=instantaneous

        self.name=f"{experiment}_{model}_{member}_{years[0]}_{years[-1]}"


        # ignore these 
        #self.tau=tau
        #self.instantaneous=instantaneous


        # built list of file names
        all_files={}
        ns=[]
        self.file_names={}
   
        if aspect=='forcing':
            for v in variables:
               
                paths=[(f"{self.data_dir}/input4mips/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
                file_names=[os.listdir(p)[0] for p in paths if os.path.exists(p)]

                full_file_names=[p+f for p in paths for f in file_names if os.path.exists(p+f)]
                #print(full_file_names)
                #full_file_names=[p+os.listdir(p)[0] for p in paths if os.path.exists(p)]
                if len(full_file_names)==0:
                    print(f"WARNING: No data available for {aspect} data combination: {self.experiment} for variable {v}. Skipping all variables of this combination.")
                    raise ValueError
                self.file_names[v]=file_names
                ns.append(len(full_file_names))
                all_files[v]=full_file_names

        elif aspect=='model':
           
            self.model=model
            self.member=member
           
            for v in variables:
                paths=[(f"{self.data_dir}/CMIP6/{self.model}/{self.member}/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
                file_names=[os.listdir(p)[0] for p in paths if os.path.exists(p)]
               
                full_file_names=[p+f for p in paths for f in file_names if os.path.exists(p+f)]
                #print(full_file_names)
                if len(full_file_names)==0:
                    print(f"WARNING: No data available for {aspect} data combination: {self.model} {self.member} {self.experiment} for variable {v}. Skipping all variables of this combination.")
                    raise ValueError
                self.file_names[v]=file_names
                ns.append(len(full_file_names))
                all_files[v]=full_file_names
        else:
            print(f"WARNING: Dataset class initialized with unknown aspect {aspect}. Please initialize with aspect being either 'forcing' or 'model'")
            exit(0)

        # checking if all desired years are available
        ns_match= (np.min(ns)==np.max(ns))
    
        # skip otherwise
        if not ns_match:
            print("WARNING: Not the same number of years for all variables!")
            exit(0) #TODO: maybe only skip pair? 

        self._num_examples=ns[0]
        self.all_files=all_files
   

        
        # read out spatial dims once
        array_per_var=[]
        for v in self.all_files:
              array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0))
          # stack variables along d dimension
        data=np.stack(array_per_var, axis=2)
        # flatten out spatial dimension
        data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))   

         
        
        # other way of getting the dimensions? -> we can just get the dimensions from the specifications but won't get faulty files
        self.data=data #TODO: if we have this anyway, do we need to redo it every time in the get_item?
        # TODO: set dimensions
        d_x=data.shape[-1]
        n=data.shape[0]
        d=data.shape[2]
        t=data.shape[1]

        self.spatial_dim=(n,t,d,d_x)

    def spatial_dim(self):
        return self.spatial_dim

    def __len__(self):
        return self._num_examples
  
    def __getitem__(self, item): # TODO figure out type signature

            # should get all years for a given experimnt
            array_per_var=[]
            for v in self.all_files:
                # stack years along n dimension
                array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0))
            # stack variables along d dimension
            data=np.stack(array_per_var, axis=2)
            # flatten out spatial dimension
            data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
            y = torch.as_tensor(data)
            # return dictionary
            D={"Data": y}
            #TODO: do it here or just od it once per initialization and then retunrn it here?
            # right now (n_years, freq, var, d_x) -> we could possibly bring together the first 2 dimensions
            return D


    def copy_to_slurm_tmp_dir(self, aspect):

        if 'SLURM_TMPDIR' in os.environ:
        #if True:
            print(f'INFO: Copying {self.name} h5 file to SLURM_TMPDIR')

            for v in self.all_files:
                if aspect=='forcing':
                    
                    path=f"{os.environ['SLURM_TMPDIR']}/input/"
                    #path = 'test/target/'

                elif aspect=='model':
                    path=f"{os.environ['SLURM_TMPDIR']}/target/"
                    #path = 'test/input/'
            
                isExist= os.path.exists(path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
                    print("INFO: The new directory on in SLURM_TMPDIR is created!")
                
                h5_path_new_in = [path + f for f in self.filenames[v]]
                
                for old_f , new_f in zip(self.all_files[v], h5_path_new_in):
                    #print(old_f, new_f)
                    shutil.copyfile(old_f, new_f)
                
                self.all_files[v]=h5_path_new_in
                self.data_dir=os.environ["SLURM_TMPDIR"]

class Causalpaca_HdF5_FastSingleDataset_Emulator(Causalpaca_HdF5_SingleDataset_Emulator):

    def __init__(self,
                 normalizer: Optional[Dict[str, Callable]] = None,
                 transform: Optional[AbstractTransform] = None,
                 write_data: bool = True,
                 reload_if_exists: bool = True,
                 *args, **kwargs):


        super().__init__(*args, **kwargs)

        
        pkwargs = dict(
            normalizer=normalizer,
            transform=transform,
           
        )
       
        #self.input_transform = input_transform
        #self.output_transform = output_transform
        self.transform=transform
        self.normalizer=normalizer
        ending = f"{self.aspect}.npz" 

        processed_fname=get_processed_fname(data_dir=self.data_dir, name=self.name, **pkwargs, ending=ending)
       
        # if everything preprocessed already existent, reload it
        if os.path.isfile(processed_fname) and reload_if_exists:
    
            self.data = self._reload_data(processed_fname)
        
        # write both if not yet existent
        else:
            self.data = self._preprocess_h5data(normalizer)
            if write_data:
                print("INFO: Writing preprocessed data.")
                self._write_data(fname=processed_fname, data=self.data)
                
    def _reload_data(self, fname):
        print(f'INFO: Reloading from {fname}.')
        try:
            in_data = np.load(fname, allow_pickle=True)
        except zipfile.BadZipFile as e:
            print(f"WARNING: {fname} was not properly saved or has been corrupted.")
            raise e
      
        return in_data

    def _write_data(self, fname, data):

        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if isinstance(data, dict) or isinstance(data, np.ndarray):
            try:
                np.savez_compressed(fname, **data) if isinstance(data, dict) else np.savez_compressed(fname, data)
            except OverflowError as e:
                print(f"WARNING: OverflowError -> {fname} will been stored as a pickle.\n{e}")
                with open(fname, "wb") as fp:
                    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
            except PermissionError as e:
                print(f"WARNING: Tried to cache data to {fname} but got error {e}. "
                            f"Consider adjusting the permissions to cache the data and make training more efficient.")
        else:
            raise ValueError(f"Data has type {type(data)}")


    def _preprocess_h5data(self,
                           normalizer: Optional[Dict[str, Callable]] = None,
                           ):
        # create data array
        # should get all years for a given experimnt
        array_per_var=[]
        for v in self.all_files:
            # stack years along n dimension
            array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0))
        # stack variables along d dimension
        data=np.stack(array_per_var, axis=2)
        # flatten out spatial dimension
        data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
        #= torch.as_tensor(data)

        # normalize
        if normalizer is not None:
            data = normalizer(data)

        # transform it
        if self.transform is not None and self.transform.save_transformed_data:
            data = self.transform.batched_transform(data)

        return_data = {
                'Data': data,
            }


        return return_data

    def copy_to_slurm_tmp_dir(self, aspect): #TODO: figure out if we inherit that or pass it
        pass

    # TODO: CHANGE AROUND
    def __getitem__(self, index) -> (Dict[str, np.ndarray]):

       
        if isinstance(self.data['Data'], np.ndarray) and len(self.data['Data'].shape) == 0:
            # Recover a nested dictionary of np arrays back, i.e. transform np.ndarray of object dtype into dict
            self.data['Data'] = self.Data['data'].item()

        if isinstance(self.data['Data'], np.ndarray):
            X = torch.from_numpy(self.data['Data']).float()
        else:
            #TODO: not sure what the else is here
            X = {k: torch.from_numpy(v[index]).float() for k, v in self.data['Data'].items()}

        if self.transform is not None and not self.transform.save_transformed_data:
            X = self.transform.transform(X)

        X = {'Data': X,
             }

        return X  

def get_processed_fname(data_dir: str, name: str, ending='.npz', **kwargs):
    processed = f"{data_dir}/{name}"
    for k, v in kwargs.items():
        if v is None:
            tr_name = 'No'
        elif isinstance(v, str):
            tr_name = v.upper()
        elif isinstance(v, NormalizationMethod) or isinstance(v, AbstractTransform):
            tr_name = v.__class__.__name__
        else:
            try:
                tr_name = v.__qualname__.replace('.', '_')
            except Exception as e:
                print(e)
                name = "?"
        processed += f'_{tr_name}_{k}'
    return processed + f'{ending}'



if __name__ == '__main__':

        experiments=["ssp119","ssp460"]
        variables=["CO2_em_anthro", "pr", "tas"]
        years=[2020, 2030, 2040]
        batch_size=2


        ds=Causalpaca_HdF5_Dataset(experiments=experiments, variables=variables, years=years, load_h5_into_mem=True)
        #for x,y in ds:
        #    print(x.size, y.size)

        """
        ds_forcing=Causalpaca_HdF5_SingleDataset_Forcing_Emulator(experiment="ssp119", data_dir=PROCESSED_DATA, years=years, variables=["CO2_em_anthro"], freq="mon", nom_res="50_km")
        print("got forcing")
        ds_model=Causalpaca_HdF5_SingleDataset_Model_Emulator(experiment="ssp126", variables=["pr", "tas"], model="NorESM2-LM", member="r1i1p1f1", data_dir=PROCESSED_DATA, years=years, freq='mon', nom_res="250_km")
        print("got model")
        for y in ds_forcing:
            print("forcing")
            print(y.shape)
            break

        for x in ds_model:
            print("model")
            print(x.shape)
            break
        """

        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        print("got data loder")
        print(len(ds))
        print(len(ds_loader))
        i=0
        for x,y in ds_loader:
            i+=1
            print("got item ")
            print(x.size(),y.size())
            if i==10:
                break