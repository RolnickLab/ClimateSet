import torch
import numpy as np
import os
from mother_data.utils.constants import VAR_SOURCE_LOOKUP
from data_paths import PROCESSED_DATA
from mother_data.utils.helper_funcs import get_keys_from_value
import h5py
from torch.utils.data import  DataLoader

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
            nom_res: str = "50_km",
            tau : int = 4,
            instantaneous:bool = False,
            target: str = 'emulator' # other choice is 'causal' leading to a different choice of base dataset / sampling
        ):

        self._name=name
        self.years=years
        self.n_years =len(years)
        self.experiments=experiments
        print("experiments", self.experiments)
        self.variables=variables
        self.models=models
        self.ensemble_members=ensemble_members
        self.freq = freq
        self.nom_res = nom_res
        self.tau=tau
        self.instantaneous=instantaneous

        print("self.data dir ", data_dir)
        if data_dir is None:
            print("data dir is none")
            print("PROCESSED DATA", PROCESSED_DATA)
            self.data_dir=PROCESSED_DATA
            print(self.data_dir)
        else:
            self.data_dir=data_dir


        self.load_h5_into_mem=load_h5_into_mem
     
        
        have_forcing=False
        have_model=False   
        self.model_variables = []
        self.forcing_variables = []

        for v in variables:

           
            t = get_keys_from_value(VAR_SOURCE_LOOKUP, v.replace("_fake", "")) #TODO: remove 'fake', introduced for debugging with incomplete data
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
        print("model vars", self.model_variables)
        print("forcing vars", self.forcing_variables)
        # dset kwargs
        dset_kwargs = dict(
            data_dir=self.data_dir,
            years=self.years,
            freq=self.freq,
            nom_res=self.nom_res,
            tau=self.tau,
            instantaneous=self.instantaneous
        ) 

     

        # might be deprecated
        self.dataset_index_to_sub_dataset_forcing: Dict[int, Tuple[int, int]] = dict()
        self.dataset_index_to_sub_dataset_model: Dict[int, Tuple[int, int]] = dict()

        self.index_to_samples_model:  Dict[int,Tuple[int, int]] = dict()
        dataset_size_forcing = 0 # TODO: check up if forcing data n matches model n 
        dataset_size_model = 0

        if self.load_h5_into_mem:
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
                dset_class_f = Causalpaca_HdF5_SingleDataset_Forcing_Causal
                dset_class_m = Causalpaca_HdF5_SingleDataset_Forcing_Causal

        self.h5_dsets_forcing= [] # create list of dset objects
        self.h5_dsets_model = []


        if have_forcing:
            # onne forcing dataset per experiment
            for exp_num_forcing, exp in enumerate(self.experiments): # have appropriate years 

                # we want one h5_dset per year
                print("exp", exp, "exp num forcing" , exp_num_forcing)
                exp_h5_dset_forcing=dset_class_f(exp, self.forcing_variables, **dset_kwargs)

                n_samples =len(exp_h5_dset_forcing)
                # read out some stuff? 
                
                # I am assuming this is depricated...
                """
                for h5_file_idx in range(n_samples):
                    # key should be experiment index
                    #
                    self.dataset_index_to_sub_dataset_forcing[h5_file_idx+dataset_size_forcing]= (exp_num_forcing, h5_file_idx)

                
              
                   # self.h5_dsets_focring list of forcing ds per experiment -> indexable by experiment subset = all years for that experiment
                # TODO how exactly does this look like?
                print("forcing index to sub dataset")
                print(self.dataset_index_to_sub_dataset_forcing)
                """
                self.h5_dsets_forcing.append(exp_h5_dset_forcing)
                dataset_size_forcing+= n_samples

            self.dataset_size_forcing=dataset_size_forcing

        if have_model:
            start=0
            end=0
            for iexp, exp in enumerate(self.experiments):
                data_experiment=False # flag meaning we have data available for the experiment
                for imod, model in enumerate(self.models):
                    for imem, member in enumerate(self.ensemble_members):
                   #

                        file_num_model=imod+imem
                        # we want one h5_dset per year
                        #print("model", model, "exp", exp, "member", member, "file num mebmber", file_num_model)

                        exp_h5_dset_model=dset_class_m(model, exp, member, self.model_variables, **dset_kwargs)
                        n_samples =len(exp_h5_dset_model)
                         
                        if n_samples==0:
                            print(f"wARNING: No data available for pairing: model {model} member {member} exp {exp}. Skipping.")
                            continue
                        else: 
                            print(f"Available! {model, member, exp}")
                        print("n_samples", n_samples)
                        # to allow sampling build dict from number experiment to all possible samples (over modles and ensemble members)
                        """" assuming this is deprecated
                        # read out some stuff? 
                        for h5_file_idx in range(n_samples):
                            self.dataset_index_to_sub_dataset_model[h5_file_idx+dataset_size_model]= (file_num_model, h5_file_idx)
                        """
                        dataset_size_model+= n_samples
                        self.h5_dsets_model.append(exp_h5_dset_forcing)
                        end+=1
                        data_experiment=True
                if data_experiment:
                    self.index_to_samples_model[iexp] = (start,end) # start -> how far did we come so far? 
                    start+=(imem+imod)
                    
            self.dataset_size_model=dataset_size_model
            print("experiment to samples index dict")
            print(self.index_to_samples_model)
            print("lenght datasets", len(self.h5_dsets_model))


            #for dset in self.h5_dsets_forcing:
            #    dset.copy_to_slurm_tmp_dir()

    @property
    def name(self):
        return self.name.upper()



    def __str__(self):
        s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
        return s

    def __len__(self):
        return self.dataset_size

    
    def __getitem__(self, item): ## add type signature

        # load forcing for a random experiment, load all years

        # for emulator model part, choose random model member for same experiments and load all years


        # bathing should happen automatically 
        which_h5, h5_index = self.dataset_index_to_sub_dataset[item]
        raw_Xs, raw_Ys = self.h5_dsets[which_h5][h5_index]

        if self._load_h5_into_mem:
            X = raw_Xs
            Y = raw_Ys
        
        else:
            # possabilty to apply transforms and normalizations 
            # TODO
            # from numpy to tensor
            X = raw_Xs
            Y = raw_Ys
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
            tau : int, # not relevant for emulator part
            instantaneous : bool # not relevant for emulator part
            ):
       

        self.experiment=experiment
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
            paths=[(f"{self.data_dir}/input4mips/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
            print("paths", paths)
            full_file_names=[p+os.listdir(p)[0] for p in paths if os.path.exists(p)]
            print(full_file_names)

            ns.append(len(full_file_names))


            #for p in paths:
            #    if not os.path.exists(p):
            #        paths.remove(p)
            #        print("Removing p", p)
            all_files[v]=full_file_names

        ns_match= (np.min(ns)==np.max(ns))
    
        if not ns_match:
            print("WARNING: not the same number of years for all variables!")
            exit(0)

        self._num_examples=ns[0]
        self.all_files=all_files
        print(self.all_files, self._num_examples)


    def copy_to_slurm_tmp_dir(self):

        # TODO: figure out exactly what to do

        raise NotImplementedError


    def __len__(self):
        return self._num_examples
  
    def __getitem__(self, index): # TODO figure out type signature


            # should get all years for a given experimnt
            array_per_var=[]
            for v in self.all_files:
                # stack years along n dimension
                array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0))
            # stack variables along d dimension
            data=np.stack(array_per_var, axis=2)
            # flatten out spatial dimension
            data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
            print("data", data.shape)
            y = torch.tensor(data)

            # right now (n_years, freq, var, d_x) -> we could possibly bring together the first 2 dimensions

            return y

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
            tau : int, # not relevant for emulator part
            instantaneous: bool # not relevant for emulator part
            ):

        # ok
        self.experiment=experiment
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
            print("variables", v)
            paths=[(f"{self.data_dir}/CMIP6/{self.model}/{self.member}/{self.experiment}/{v}/{self.nom_res}/{self.freq}/{y}/") for y in self.years]
            print("paths", paths)

            full_file_names=[p+os.listdir(p)[0] for p in paths if os.path.exists(p)]
            print("full, fill names", full_file_names)
            ns.append(len(full_file_names))

            #for p in paths:
            #    if not os.path.exists(p):
            #        paths.remove(p)
            #        print("Removing p", p)
            all_files[v]=full_file_names

        ns_match= (np.min(ns)==np.max(ns))
    
        if not ns_match:
            print("WARNING: not the same number of years for all variables!")
            exit(0)

        self._num_examples=ns[0]
        self.all_files=all_files
        #print(self.all_files, self._num_examples)


    def copy_to_slurm_tmp_dir(self):

        # TODO: figure out exactly what to do

        raise NotImplementedError


    def __len__(self):
        return self._num_examples
  
    def __getitem__(self, index): # TODO figure out type signature


            # should get all years for a given experimnt
            array_per_var=[]
            for v in self.all_files:
                # stack years along n dimension
                array_per_var.append(np.stack([h5py.File(f, 'r')[v] for f in self.all_files[v]],axis=0))
            # stack variables along d dimension
            data=np.stack(array_per_var, axis=2)
            # flatten out spatial dimension
            data=np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
            print("data", data.shape)
            x = torch.tensor(data)

            # right now (n_years, freq, var, d_x) -> we could possibly bring together the first 2 dimensions

            return x


if __name__ == '__main__':

        experiments=["ssp119","ssp460"]
        variables=["BC_em_anthro", "CO2_em_anthro", "pr_fake", "tas_fake"]
        years=[2020, 2030, 2040]
        batch_size=8


        ds=Causalpaca_HdF5_Dataset(experiments=experiments, variables=variables, years=years)

        #ds_forcing=Causalpaca_HdF5_SingleDataset_Forcing_Emulator(experiment="ssp119", data_dir=PROCESSED_DATA, years=years, variables=variables, freq="mon", nom_res="50_km")


        #for y in ds_forcing:
        #    print(y.shape)
        


        #ds_loader = DataLoader(ds_forcing, batch_size=batch_size, shuffle=False)