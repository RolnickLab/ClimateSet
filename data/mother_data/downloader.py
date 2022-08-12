from utils.constants import MODEL_SOURCES, VAR_SOURCE_LOOKUP
from utils.helper_funcs import get_keys_from_value


from siphon import catalog #TDS catalog
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

                experiments: [str] = ['1pctCO2', 'historical', 'piControl'], #sub-selection of ClimateBench defaul
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

                try:
                    self.model_source=MODEL_SOURCES[self.model]
                except KeyError:
                    print(f"WARNING: Model {self.model} unknown. Using default instead.")
                    self.model=next(iter(MODEL_SOURCES))
                    self.model_source=MODEL_SOURCES[self.model]
                    print('Using:', self.model)
                print('model source:', self.model_source)
                self.model_catalog=catalog.TDSCatalog(self.model_source)
                self.num_ensemble = num_ensemble
                print("Read full catalogue.")

                print("datasets", self.model_catalog.datasets)
                print("services", self.model_catalog.services)
                print("catalog refs", self.model_catalog.catalog_refs)



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
            # iterate over experiments
            for e in self.experiments:
                # iterate over number of ensemble members
                for m in range(self.num_ensemble):

                    physics = physics = 2 if experiment == 'ssp245-covid' else 1  # The COVID simulation uses a different physics setup  #TODO: taken from ClimateBench, clarify what it means

                    member = member = f"r{i+1}i1p1f{physics}"
                    print(f"Processing {member} of {experiment}...")

                    ds, res = self.get_raw_data(v,e,m)

                    outfile = f"{var}/{experiment}/{member}/{res}.nc"
                    if (not overwrite) and os.path.isfile(outfile):
                        print(f"File {outfile} already exists, skipping.")
                        continue

                    # TODO: store -> what format??






    def get_raw_data(self, variable, experiment, ensemble_member):
      """
      Inspired by https://github.com/rabernat/pangeo_esgf_demo/blob/master/narr_noaa_thredds.ipynb
      """

     # TODO:
     # search for lowest resolution possible
     # what can we already kick out? 

      # Get the relevant catalog references
      cat_refs = list({k:v for k,v in self.catalog.catalog_refs.items() if k.startswith(f"CMIP6.{get_MIP(experiment)}.NCC.NorESM2-LM.{experiment}.{ensemble_member}.day.{variable}.")}.values())
      # Get the latest version (in case there are multiple)
      print(cat_refs)
      cat_ref = sorted(cat_refs, key=lambda x: str(x))[-1]
      print(cat_ref)
      sub_cat = cat_ref.follow().datasets
      datasets = []
      # Filter and fix the datasets
      for cds in sub_cat[:]:
        # Only pull out the (un-aggregated) NetCDF files
        if (str(cds).endswith('.nc') and ('aggregated' not in str(cds))):
          # For some reason these OpenDAP Urls are not referred to as Siphon expects...
          cds.access_urls['OPENDAP'] = cds.access_urls['OpenDAPServer']
          datasets.append(cds)
      dsets = [(cds.remote_access(use_xarray=True)
                 .reset_coords(drop=True)
                 .chunk({'time': 365}))
             for cds in datasets]
      ds = xr.combine_by_coords(dsets, combine_attrs='drop')
      return ds[variable]

if __name__ == '__main__':

    downloader = Downloader()
