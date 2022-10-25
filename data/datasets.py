import torch
import numpy as np
import os
from mother_data.utils.constants import VAR_SOURCE_LOOKUP
from data_paths import PROCESSED_DATA
from mother_data.utils.helper_funcs import get_keys_from_value
import h5py
from torch.utils.data import  DataLoader
import random

class Causalpaca_HdF5_Dataset(torch.utils.data.Dataset): # TODO: where to do the training vs testing discrimination? 
    """Class for working with multiple HDF5 datasets"""

    def __init__(
            self,
            years: [int],
            experiments: [str],
            variables: [str],
            name: str = "test",
            data_dir: str = None,
            load_h5_into_mem: bool = False,
            models : [str] = ["NorESM2-LM"],
            ensemble_members: [str] = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
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
            raise NotImplementedError
            if target=='emulator':

                # given emulator, we get batch size many experiment_variables pairs sample batch_size examples form them
                # each sample contains all possible year
                dset_class_f = Causalpaca_HdF5_FastSingleDataset_Forcing_Emulator
                dset_class_m = Causalpaca_HdF5_FastSingleDataset_Model_Emulator
            elif target=='causal':

                dset_class_f = Causalpaca_HdF5_FastSingleDataset_Forcing_Causal
                dset_class_m = Causalpaca_HdF5_FastSingleDataset_Model_Causal

        else:
            if target=='emulator':
                dset_class_f = Causalpaca_HdF5_SingleDataset_Forcing_Emulator
                dset_class_m = Causalpaca_HdF5_SingleDataset_Model_Emulator
            elif target=='causal':
                print("WARNING: Causal Dataloader not yet implemented")
                raise NotImplementedError
                dset_class_f = Causalpaca_HdF5_SingleDataset_Forcing_Causal
                dset_class_m = Causalpaca_HdF5_SingleDataset_Forcing_Causal

        dataset_size_forcing = 0 # TODO: check up if forcing data n matches model n 
        dataset_size_model = 0
        self.h5_dsets_forcing= {} # create list of dset objects
        self.h5_dsets_model = {}

        if have_forcing:
            # onne forcing dataset per experiment
            for exp in self.experiments: 

                exp_h5_dset_forcing=dset_class_f(exp, self.forcing_variables, nom_res=forcing_nom_res, **dset_kwargs)
                n_samples =len(exp_h5_dset_forcing)
                # TODO: read out some stuff? 
                self.h5_dsets_forcing[exp]=exp_h5_dset_forcing
                dataset_size_forcing+= n_samples

            self.dataset_size_forcing=dataset_size_forcing

        if have_model:
            for exp in self.experiments:
               
                self.h5_dsets_model[exp]=[]
                for  model in self.models:
                    for member in self.ensemble_members:
                        
                       
                        exp_h5_dset_model=dset_class_m(model, exp, member, self.model_variables, nom_res=model_nom_res, **dset_kwargs)
                        n_samples =len(exp_h5_dset_model)
                         
                        if n_samples==0:
                            print(f"wARNING: No data available for pairing: model {model} member {member} exp {exp}. Skipping.")
                            continue
                        
                        dataset_size_model+= n_samples
                       
                        self.h5_dsets_model[exp].append(exp_h5_dset_model)
               
            #for dset in self.h5_dsets_forcing:
            #    dset.copy_to_slurm_tmp_dir()

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
        raw_Xs = forcing["Data"]

        # choose data belonging to a random model / member pair
        model_data=random.choice(self.h5_dsets_model[exp])
        raw_Ys = model_data["Data"]

        if self.load_h5_into_mem:
            X = raw_Xs
            Y = raw_Ys
        
        else:
            # possabilty to apply transforms and normalizations 
            # TODO
            X=raw_Xs["Data"]
            Y=raw_Ys["Data"]
     
        return X, Y


        def spatial_dim(self) -> Dict[str, int]:
            return sefl.h5_dsets[0].spatial_dim #TODO

        def input_dim(self) -> Dict[str, int]:
            return sefl.h5_dsets[0].input_dim #TODO
        
        def output_dim(self) -> Dict[str, int]:
            return sefl.h5_dsets[0].output_dim #TODO

        def close(self):
            for dset in self.h5_dsets:
                dset.close()
        

class Causalpaca_HdF5_SingleDataset_Forcing_Emulator(torch.utils.data.Dataset):

    def __init__(self,
            experiment : str,
            variables : [str],
            data_dir : str,
            years : [int],
            freq : str,
            nom_res : str,
            tau : int = 0, # not relevant for emulator part
            instantaneous : bool = False # not relevant for emulator part
            ):
       

        self.experiment="ssp119"#experiment #TODO: remove (debugging purpose)
        self.data_dir=data_dir
        self.variables=variables
        self.years=years
        
        self.freq=freq
        self.nom_res=nom_res

        # ignore these 
        #self.tau=tau
        #self.instantaneous=instantaneous


        # built list of file names
        all_files={}
        ns=[]
        for v in variables:
            paths=[(f"{self.data_dir}/input4mips/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
            full_file_names=[p+os.listdir(p)[0] for p in paths if os.path.exists(p)]
            ns.append(len(full_file_names))
            all_files[v]=full_file_names

        # checking if all desired years are available
        ns_match= (np.min(ns)==np.max(ns))
    
        # skip otherwise
        if not ns_match:
            print("WARNING: not the same number of years for all variables!")
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
        self.d_x=data.shape[-1]
        self.n=data.shape[0]
        self.d=data.shape[2]
        self.t=data.shape[1]
     

    def copy_to_slurm_tmp_dir(self):

        # TODO: figure out exactly what to do

        raise NotImplementedError


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

class Causalpaca_HdF5_SingleDataset_Model_Emulator(torch.utils.data.Dataset):

    def __init__(self,
            model: str,
            experiment : str,
            member : str,
            variables : [str],
            data_dir : str,
            years : [int], # ok
            freq : str,
            nom_res : str,
            tau : int = 0, # not relevant for emulator part
            instantaneous: bool = False # not relevant for emulator part
            ):

        self.experiment="ssp126"#experiment #TODO: for debugging, remove 
        self.member=member
        self.model = model
        self.data_dir=data_dir
        self.variables=variables
        self.years=years
        
        self.freq=freq
        self.nom_res=nom_res

        # ignore these 
        #self.tau=tau
        #self.instantaneous=instantaneous

        # TODO: set dimensions
        self.spatial_dim = None
        self.output_dim = None
        self.input_dim = None

        # built list of file names
        all_files={}
        ns=[]
        for v in variables:
          
            paths=[(f"{self.data_dir}/CMIP6/{self.model}/{self.member}/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
            full_file_names=[p+os.listdir(p)[0] for p in paths if os.path.exists(p)]
            ns.append(len(full_file_names))
            all_files[v]=full_file_names

        ns_match= (np.min(ns)==np.max(ns))
    
        if not ns_match:
            print("WARNING: not the same number of years for all variables!")
            exit(0)

        self._num_examples=ns[0]
        self.all_files=all_files
   

    def copy_to_slurm_tmp_dir(self):

        # TODO: figure out exactly what to do

        raise NotImplementedError


    def __len__(self):
        return self._num_examples
  
    def __getitem__(self, item): # TODO figure out type signature

            array_per_var=[]
            for v in self.all_files:
                array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0)) #TODO: reverse cheating
            # stack variables along d dimension
            data=np.stack(array_per_var, axis=2)
            # flatten out spatial dimension
            data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
            #print("data", data.shape)
            x = torch.as_tensor(data)
            D={"Data": x}
            # right now (n_years, freq, var, d_x) -> we could possibly bring together the first 2 dimensions
            return D


if __name__ == '__main__':

        experiments=["ssp119","ssp460"]
        variables=["CO2_em_anthro", "pr", "tas"]
        years=[2020, 2030, 2040]
        batch_size=2


        ds=Causalpaca_HdF5_Dataset(experiments=experiments, variables=variables, years=years)
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