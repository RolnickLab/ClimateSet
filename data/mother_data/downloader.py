from utils.constants import MODEL_SOURCES, VAR_SOURCE_LOOKUP, GRIDDING_HIERACHY, VAR_RES_MAPPING_PATH
from utils.helper_funcs import get_keys_from_value, runcmd, get_MIP, get_lowest_entry


from siphon import catalog #TDS catalog
import pandas as pd
import xarray as xr
import pandas as pd

import os.path
overwrite = False

# TODO argparse with parameter where data should be stored

# Downloads data of two types:

# maybe one class "downloader"
# two subclasses for input / output

# Input of climate models (different source)
    # "really raw input": normal input variables
    # "model raw input": CO2 mass and other specific variables: assumptions within the model, depending on the SSP path
    # (different preprocessing, normalize (CO2 mass minus baseline), see ClimateBench)

# Predictions / output of climate models (different source)

# Storage:
# Raw-raw data can be deleted after preprocessing (preprocessing always with highest resolution)

# Resolution:
# highest possible resolution [??]

# class Downloader
# source: link whatever
# storage: where to store (data_paths)
# params: mother_params
# ATTENTION: download always highest resolution, res params are only used later on during res_preprocesser

# TODO: where to store sources links for data

# no returns, but communicates to user where stuff was stored and if sucessful
# hand over exceptions if there are any problems, or print commands


class Downloader:

    def __init__(self,
                model: str = "NorESM2-L", #defaul as in ClimateBench

                experiments: [str] = ['1pctCO2', 'ssp370', 'historical', 'piControl'], #sub-selection of ClimateBench defaul
                vars: [str] = ['tas', 'pr'],
                num_ensemble: int = 1, #number of ensemble members

                ):

                # TODO: check if model is supported
                self.model=model
                self.experiments=experiments

                # assign vars to either target or raw source
                self.raw_vars=[]
                self.model_vars=[]
                for v in vars:
                    t=get_keys_from_value(VAR_SOURCE_LOOKUP, v)
                    if t=='model':
                        self.model_vars.append(v)
                    elif t=='raw':
                        self,raw_vars.append(v)

                    else:
                        print(f"WARNING: unknown source type for var {v}.")

                print(f"Raw variables to download: {self.raw_vars}")
                print(f"Model prediced vars to download: {self.model_vars}")


                # Mapping from var to lowest resolution available using the .csv
                var2res=pd.read_csv(VAR_RES_MAPPING_PATH, delimiter=',',dtype=str)
                # only extract relevant vars
                indxs=var2res['variable_id'].isin(self.model_vars)
                available_res=var2res[indxs]
                # taking care of how resolutions are stored in csv -> list of strings
                available_res['resolutions']=available_res['table_id(s)'].apply(lambda l: l.split(',')).apply(lambda l: [i.strip() for i in l])
                # make indexable by var
                self.var2res=available_res.set_index(available_res['variable_id'])

                try:

                    # TODO: switch to modelsoruce.csv (add url there)
                    self.model_source_url=MODEL_SOURCES[self.model]["url"]
                    self.model_source_center=MODEL_SOURCES[self.model]["center"]
                except KeyError:
                    print(f"WARNING: Model {self.model} unknown. Using default instead.")
                    self.model=next(iter(MODEL_SOURCES))
                    self.model_source_url=MODEL_SOURCES[self.model]["url"]
                    self.model_source_center=MODEL_SOURCES[self.model]["center"]
                    print('Using:', self.model)
                print('model source url:', self.model_source_url)
                self.catalog=catalog.TDSCatalog(self.model_source_url)
                print("Read full catalogue.\n")


                self.num_ensemble = num_ensemble

                # TODO: create folder hierachy / check if existent make new if not

                #TODO: more checkups?

    def download_raw(self):
        raise NotImplementedError



    def download_from_model(self):
        """
        Searches for all filles associated with the respected variables and experiment that we want to consider.
        Attempts to download the highest resolution available.
        """


        # iterate over respective vars
        for v in self.model_vars:
            print(f"Downloading data for variable: {v} \n \n ")
            # iterate over experiments
            for e in self.experiments:
                print(f"Downloading data for experiment: {e}\n")
                self.get_raw_data(v,e)






    def get_raw_data(self, variable: str, experiment: str):
        """
        Inspired by: //TODO insert reference (liken in ClimateBench)


        """
        # catalog_refs = list of references in a catalogue
        # .follow() follows a reference and returns the new catalog

        # Step 1: map experiment to MIP
        cat=self.catalog.catalog_refs[get_MIP(experiment)].follow()

        # Step 2: access model and experiment
        cat=cat.catalog_refs[self.model_source_center].follow().catalog_refs[self.model].follow().catalog_refs[experiment].follow()

        # Step 3: iterate over ensemble members
        for i in range(self.num_ensemble):

            physics = 2 if experiment == 'ssp245-covid' else 1  # The COVID simulation uses a different physics setup  #TODO: taken from ClimateBench, clarify what it means

            member = f"r{i+1}i1p1f{physics}"
            print(f"Var to download: {variable}\n Processing {member} of {experiment}...")

            new_c=cat.catalog_refs[member].follow()

            # Step 4: search for lowest resolution and check if variable availabe

            resolutions=list(new_c.catalog_refs)

            print("available resolutions:")

            print(resolutions)

            hierachy=self.var2res['resolutions'][variable]

            print(f'choosing resolution according to hierachy: \n', hierachy)
            #print('new c', new_c.name, new_c.value)
            res, new_c =self.get_resolution(catalog=new_c, resolutions=resolutions, hierachy=hierachy, variable=variable)
            if res is None:
                break

            # Step 5: choose gridding
            print('available griddings:')
            griddings=new_c.catalog_refs
            print(griddings)
            grid=get_lowest_entry(griddings, GRIDDING_HIERACHY)
            if grid is None:
                print("Something went wrong... No desired gridding found. Skipping.")
                break
            print(f"proceeding with gridding: {grid}")

            new_c=new_c.catalog_refs[grid].follow().catalog_refs[0].follow() #TODO: chose 0 here for latet i guess but there are other options 'files, latest'

            sub_cats=new_c.datasets
            print('available files')
            print(sub_cats)

            datasets=[]

            for cds in sub_cats[:]:
                # Only pull out the (un-aggregated) NetCDF files
                if (str(cds).endswith('.nc') and ('aggregated' not in str(cds))):
                        datasets.append(cds)
                dsets = [(cds.remote_access(use_xarray=True)
                                .reset_coords(drop=True)
                                .chunk({'time': 365}))
                                for cds in datasets]

                ds = xr.combine_by_coords(dsets, combine_attrs='drop')
                print(ds[variable])

                outfile = f"{variable}/{experiment}/{member}/{res}.nc"
                if (not overwrite) and os.path.isfile(outfile):
                    print(f"File {outfile} already exists, skipping.")
                    continue


            #TODO: handle ensembel members (efficiently / how to build dataset /direct combining?)
            # TODO: create folder hierachy / check if existent make new if not
            #ds.to_netcdf(outfile

    def get_resolution(self, catalog: catalog.TDSCatalog, resolutions: [str], hierachy: [str], variable: str):

        checking_res=True
        while checking_res:
            res=get_lowest_entry(resolutions, hierachy)

            if res is None:
                print("Something is wrong... Now matching resolution found. Skipping.")
                return None, _

            print(f"Ateempting to proceeding with temporal resolution of: {res}")

            try:
                catalog=catalog.catalog_refs[res].follow().catalog_refs[variable].follow()
                checking_res=False

            except KeyError:
                print(f"Resolution {res} not available for variable {variable}. \n Choosing a different resolution.")
                catalog=catalog

                if len(resolutions)>1:
                    resolutions.remove(res)
                else:
                    print(f"No available resolution found. Skipping.")
                    return None, _
        return res, catalog

if __name__ == '__main__':
    test_mother=True

    if test_mother:
        print('testing mother')
        downloader = Downloader()
        downloader.download_from_model()

    else:
        print('debugging')
        #catalog=catalog.TDSCatalog("https://dap.ceda.ac.uk/thredds/catalog/badc/cmip6/data/CMIP6/catalog.xml")
        catalog=catalog.TDSCatalog("https://dap.ceda.ac.uk/thredds/catalog/badc/cmip6/data/CMIP6/catalog.xml")
        print("read catalog")
        print("datasets", catalog.datasets)
        print(catalog.services)
        print(catalog.catalog_refs)

        """
        for k,v in catalog.catalog_refs.items():
            print(k,v)

            print(catalog.catalog_refs[k])
            new_c=catalog.catalog_refs[k].follow()
            print(new_c.catalog_refs)
        """
        variable='tas'
        ensemble_member="r1i1p1f1"
        res='day'
        for e in ['ssp370', 'historical', 'piControl']:

            # accessing experiment datasets
            new_c=catalog.catalog_refs[get_MIP(e)].follow().catalog_refs['NCC'].follow().catalog_refs['NorESM2-LM'].follow().catalog_refs[e].follow().catalog_refs[ensemble_member].follow()
            # 0 for version or files or latest? 
            res_c=new_c.catalog_refs[res].follow().catalog_refs[variable].follow().catalog_refs['gn'].follow().catalog_refs[0].follow()
            print('followed experiment + member + var + res')


            sub_cats=res_c.datasets
            datasets=[]

            for cds in sub_cats[:]:
              # Only pull out the (un-aggregated) NetCDF files
              if (str(cds).endswith('.nc') and ('aggregated' not in str(cds))):
                # For some reason these OpenDAP Urls are not referred to as Siphon expects...
                #cds.access_urls['OPENDAP'] = cds.access_urls['OpenDAPServer']
                datasets.append(cds)
            dsets = [(cds.remote_access(use_xarray=True)
                       .reset_coords(drop=True)
                       .chunk({'time': 365}))
                   for cds in datasets]
            ds = xr.combine_by_coords(dsets, combine_attrs='drop')
            print(ds[variable])


            """
            print(e)
            string=f"{get_MIP(e)}/NCC/NorESM2"#.NorESM2-LM"#.{e}"#".{ensemble_member}.day.{variable}."
            print(string)
            cat_refs = list({k:v for k,v in catalog.catalog_refs.items() if k.startswith(string)}.values())
            print(cat_refs)
            cat_ref = sorted(cat_refs, key=lambda x: str(x))[-1]
            print(cat_ref)
            sub_cat = cat_ref.follow().datasets
            print(sub_cat)
            datasets = []

            """
