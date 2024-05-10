
import os 
import glob
import numpy as np
import xarray as xr
from typing import List
from emulator.src.utils.utils import get_years_list
import matplotlib.pyplot as plt
import os



LON = 96
LAT = 144
NUM_LEVELS = 1
SEQ_LEN = 12
CMIP6_NOM_RES = '250_km'
CMIP6_TEMP_RES = 'mon'
DATA_DIR = '/home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/Climateset_DATA' 

format='pdf'
#climate_models=['CESM2-WACCM','MPI-ESM1-2-HR','TaiESM1','EC-Earth3-Veg']
climate_models=os.listdir('/home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/Climateset_DATA/outputs/CMIP6')

climate_models.remove("EC-Earth3-Veg-LR") # problem with historical data! #TODO
climate_models.remove("BCC-CSM2-MR") # weird historical mean
print("models: ", climate_models)
#scenarios=["ssp126","ssp245","ssp370", "ssp585"]  # needs to be sorted!
scenarios=["ssp585", "ssp370", "ssp245", "ssp126"]
print(scenarios)
median_index=int(len(scenarios)/2)

years="2015-2100"
historical_years="1960-1990"
var='tas'
num_ensembles=1
data_save_dir="data"
plot_save_dir="plots"

# substracting historical data
historical_data = True
sort_temperature = True

plot_years_interval=10
alpha_value=0.3
colors_models=[f'C{i}' for i in range(len(climate_models))]
#colors_scenarios=[f'C{i}' for i in range(len(scenarios))]
#colors_scenarios=["yellowgreen","gold","darkorange","crimson"]
colors_scenarios=["crimson", "darkorange", "gold", "yellowgreen"]
cmap="RdYlGn"
#colors_scenarios=plt.cm.RdYlGn(np.linspace(0,1,len(scenarios)))
colors_models=plt.cm.RdYlGn(np.linspace(0,1,len(climate_models)))

# years to years list
get_years=get_years_list(years, give_list=True)
get_years_historical=get_years_list(historical_years, give_list=True)

concat = True
missing_files=[]
missing_files_historical=[]

def load_into_mem( paths: List[List[str]], seq_len=12, historical=False): #-> np.ndarray():
            concat = True
            array_list =[]
            missing_files=[]

            for vlist in paths:
                #print("vlist")
                #print(vlist)
              
                try:
                    temp_data = xr.open_mfdataset(vlist, concat_dim='time', combine='nested').compute() #.compute is not necessary but eh, doesn't hurt
                    temp_data = temp_data.to_array().to_numpy()
                except:
                    missing_files.append(vlist)
                    concat=False
                    continue
                 
                array_list.append(temp_data)
            if concat:
                temp_data = np.concatenate(array_list, axis=0)

                if historical:
                     temp_data = temp_data.reshape(1, -1, seq_len, LON, LAT) 
                else:  
                    temp_data = temp_data.reshape(len(scenarios), -1, seq_len, LON, LAT) # num_vars, num_scenarios*num_remainding_years, seq_len,lon,lat)
                temp_data = np.expand_dims(temp_data, 0) 
                
                return temp_data # (1, num_scenarios, years,seq_len, 96, 144)
            else: 
                print("missing files", missing_files)
                return np.asarray([])


# get ensemble dir list
models = []
# historical
models_historical = []


root_dir_original = os.path.join(DATA_DIR, "outputs/CMIP6")

for model in climate_models:
        root_dir = os.path.join(root_dir_original, model)

        if num_ensembles == 1:
            ensembles = os.listdir(root_dir)
            ensemble_dir =  [os.path.join(root_dir, ensembles[0])] # Taking first ensemble member
        else:
            print("Multiple ensembles", num_ensembles)
            ensemble_dir = []
            ensembles = os.listdir(root_dir)
            for i,folder in enumerate(ensembles):
                ensemble_dir.append(os.path.join(root_dir, folder)) # Taking multiple ensemble members
                if i==(num_ensembles-1):
                    break

        model_member=[]
        model_member_historical=[]    
        for i,em in enumerate(ensemble_dir):
                # load per ensemble member

                # reloadi if existenti
                file_name=model+"_"+str(i)+"_"+"_".join(map(str,scenarios))+"_"+years+".npy"
               
                print(file_name)
                
                if os.path.isfile(os.path.join(data_save_dir,file_name)): # we first need to get the name here to test that...
                    print("path exists, reloading")
                    raw_data = np.load(os.path.join(data_save_dir,file_name))
                    model_member.append(raw_data)
                else:
                    
                    output_nc_files=[]
                
                    for exp in scenarios:
                        
                            for y in get_years:
                                    # we only have one ensemble here
                                    var_dir = os.path.join(em, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}') 
                                    files = glob.glob(var_dir + f'/*.nc', recursive=True)
                                    if len(files)==0:
                                        print("No files for this climate model, ensemble member, var, year ,scenario:", model,em.split('/')[-1],var, y, exp)
                                        print("Exiting! Please fix the data issue.")
                                        missing_files.append(f"{model} {em.split('/')[-1]} {exp} {var} {y}")
                                        concat=False
                                        continue
                                    
                                    else:
                                        output_nc_files += files
                    raw_data = load_into_mem(output_nc_files)
                    raw_data=np.expand_dims(raw_data,axis=0)

                    np.save(os.path.join(data_save_dir, file_name), raw_data)
                    model_member.append(raw_data)

                if historical_data:
                    # reloadi if existenti
                    file_name=model+"_"+str(i)+"_"+"historical"+"_"+historical_years+".npy"
               
                    print(file_name)
                
                    if os.path.isfile(os.path.join(data_save_dir,file_name)): # we first need to get the name here to test that...
                        print("path exists, reloading")
                        raw_data = np.load(os.path.join(data_save_dir,file_name))
                        model_member_historical.append(raw_data)
                    else:
                        print("creating file")
                        output_nc_files_historical=[]
                        exp="historical"
                        
                        for y in get_years_historical:
                                    # we only have one ensemble here
                                    var_dir = os.path.join(em, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}') 
                                    files = glob.glob(var_dir + f'/*.nc', recursive=True)
                                    if len(files)==0:
                                        print("No files for this climate model, ensemble member, var, year ,scenario:", model,em.split('/')[-1],var, y, exp)
                                        print("Exiting! Please fix the data issue.")
                                        missing_files_historical.append(f"{model} {em.split('/')[-1]} {exp} {var} {y}")
                                        concat=False
                                        continue
                                    
                                    else:
                                        output_nc_files_historical += files
                        raw_data = load_into_mem(output_nc_files_historical, historical=True)
                        raw_data=np.expand_dims(raw_data,axis=0)
                        np.save(os.path.join(data_save_dir,file_name), raw_data)
                        model_member_historical.append(raw_data)
        if concat:
            model_member_data = np.concatenate(model_member, axis=0)
            if historical_data:
                model_member_data_historical = np.concatenate(model_member_historical, axis=0)
        else:
            model_member_data=[]
            if historical_data:
                model_member_data_historical=[]
        models.append(model_member_data)
        if historical_data:
            models_historical.append(model_member_data_historical)
print("Len of models list", len(models))

if concat:
    data = np.concatenate(models, axis=0)
    if historical_data:
        print([mh.shape for mh in models_historical])
        data_historical = np.concatenate(models_historical, axis=0)
else:
    print("Mising files:")
    print(missing_files)
    if historical_data:
        print("Missing historical files:")
        print(missing_files_historical)
    print("Cannot concatenate, exting.")
    exit(0)
    
deg2rad=True    
if deg2rad:
    weights=np.cos((np.pi * np.arange(data.shape[-1]))/180)
else:
    weights=np.cos(np.arange(data.shape[-1]))
# latitude weigts
# in climax weights are normalized first
weights = weights / weights.mean()

# we want the shape (climate_models, num_ensembles, exp, years, seq_len, lon, lat)
print("data", data.shape)

# average over seq_len
data = np.mean(data, axis = -3)
# latitude weigts
deg2rad=True    
if deg2rad:
    weights=np.cos((np.pi * np.arange(data.shape[-1]))/180)
else:
    weights=np.cos(np.arange(data.shape[-1]))

# in climax weights are normalized first
weights = weights / weights.mean()

# average over ln lat (weighted)
data = weights*data
data = np.mean(data, axis = (-1,-2))
print("data", data.shape)


# average over ensemble members
data = np.mean(data, axis = 1)
print("data", data.shape)

if historical_data: 
    print("data_historical")
    print(data_historical.shape)
    # average over seq_len
    data_historical = np.mean(data_historical, axis = -3)
    # average over ln lat (weighted)
    data_historical = weights*data_historical
    data_historical = np.mean(data_historical, axis = (-1,-2))
    print("data historical", data_historical.shape)
    # average over ensemble members
    data_historical = np.mean(data_historical, axis = 1)
    print("data_historical", data_historical.shape)
    # shape should be (models,1,years)
    data_historical_mean = np.expand_dims(np.mean(data_historical, axis=-1),-1)
    print("historical mean shape")
    print(data_historical_mean.shape)
    print(data_historical_mean)
    # now substracting 
    print("substracting")
    data = data-data_historical_mean
    print("final data shape", data.shape)

# shape: (models, scenarios, years)

# plot: x - years, y - variable (global avg)
x = np.arange(len(get_years))
plot_years=np.arange(5, len(get_years)+1, plot_years_interval)
fig, ax = plt.subplots()
# plot 1: lines are climate models, variance are ssps
# option a: pick first median and last ssp

# sort data according to temperature mean over ssp
if sort_temperature:
    
    # take median ssp mean over time
    #median_ssp = data[:,median_index,-1] # option to mean over time or take last time step
    print("mean ssp")
    mean_ssp=np.mean(data,axis=1)
    print(mean_ssp.shape)
    mean_ssp=mean_ssp[:,-1]
    print(mean_ssp.shape)
    print(mean_ssp)
    indices = np.argsort(mean_ssp)
    indices = indices[::-1] # reverse tor order in legend (hot to cold)

    print("new indices")
    print(indices)
    print("sorted models")
    climate_models = [climate_models[i] for i in indices]
    print(climate_models)
else:
    indices=np.arange(len(climate_models))


for e,(i,m) in enumerate(zip(indices,climate_models)):
    # pick median ssp, min, max 
    print(m)
    
    mean_ssp=data[i][median_index]
    max_ssp=data[i][-1]
    min_ssp=data[i][0]
    ax.plot(x, mean_ssp, color=colors_models[e], label=m)
    plt.fill_between(x, min_ssp, max_ssp,color=colors_models[e], alpha=alpha_value)
    #ax.plot(x, mean_ssp, cmap=cmap, m)
    #plt.fill_between(x, min_ssp, max_ssp,cmap=cmap, alpha=alpha_value)


ax.set_xticks(plot_years,[get_years[i] for i in plot_years])
ax.set_xlabel("Year A.D.")
if historical_data:
    ax.set_ylabel(f"$\Delta$ Temperature [K]")
else:
    ax.set_ylabel(var)
ax.legend()
plt.margins(x=0)
fig.savefig(f'{plot_save_dir}/model_scenario_variance_delta_historical_{historical_data}.{format}')

fig.clf()
fig, ax = plt.subplots()

# option b: compute min, max, mean ssp values
for e,(i,m) in enumerate(zip(indices,climate_models)):
    # pick median ssp, min, max 
    mean_ssp=np.mean(data[i],axis=0)
    max_ssp=np.max(data[i],axis=0)
    min_ssp=np.min(data[i],axis=0)
    ax.plot(x, mean_ssp, color=colors_models[e], label=m)
    plt.fill_between(x, min_ssp, max_ssp, linewidth=0,color=colors_models[e], alpha=alpha_value)


ax.set_xticks(plot_years,[get_years[i] for i in plot_years])
ax.set_xlabel("Year A.D.")
if historical_data:
    ax.set_ylabel(f"$\Delta$ Temperature [K]")
else:
    ax.set_ylabel(var)
ax.legend()
plt.margins(x=0)
fig.savefig(f'{plot_save_dir}/model_scenario_variance_compute_min_max_delta_historical_{historical_data}.{format}')


fig.clf()
fig, ax = plt.subplots()

# option c: compute mean + std ssp values
for e,(i,m) in enumerate(zip(indices,climate_models)):
    mean_ssp=np.mean(data[i],axis=0)
    std_ssp=np.std(data[i],axis=0)
    max_ssp=mean_ssp+std_ssp
    min_ssp=mean_ssp-std_ssp
    ax.plot(x, mean_ssp, color=colors_models[e], label=m)
    plt.fill_between(x, min_ssp, max_ssp, linewidth=0,color=colors_models[e], alpha=alpha_value)


ax.set_xticks(plot_years,[get_years[i] for i in plot_years])
ax.set_xlabel("Year A.D.")
if historical_data:
    ax.set_ylabel(f"$\Delta$ Temperature [K]")
else:
    ax.set_ylabel(var)
ax.legend()
plt.margins(x=0)
fig.savefig(f'{plot_save_dir}/model_scenario_variance_compute_std_delta_historical_{historical_data}.{format}')


fig.clf()
fig, ax = plt.subplots()

# plot 2: lines are ssps, variance are climate models

# option a: compute mean and std
for i,m in enumerate(scenarios):
    mean_ssp=np.mean(data[:,i,:],axis=0)
    std_ssp=np.std(data[:,i,:],axis=0)
    max_ssp=mean_ssp+std_ssp
    min_ssp=mean_ssp-std_ssp
    ax.plot(x, mean_ssp, color=colors_scenarios[i], label=scenarios[i])
    plt.fill_between(x, min_ssp, max_ssp, linewidth=0,color=colors_scenarios[i], alpha=alpha_value)

ax.set_xticks(plot_years,[get_years[i] for i in plot_years])
ax.set_xlabel("Year A.D.")
if historical_data:
    ax.set_ylabel(f"$\Delta$ Temperature [K]")
else:
    ax.set_ylabel(var)
ax.legend()
plt.margins(x=0)
fig.savefig(f'{plot_save_dir}/scenario_model_variance_compute_std_delta_historical_{historical_data}.{format}')

fig.clf()
fig, ax = plt.subplots()

# option a: compute mean and std
for i,m in enumerate(scenarios):
    mean_ssp=np.mean(data[:,i,:],axis=0)
    max_ssp=np.max(data[:,i,:],axis=0)
    min_ssp=np.min(data[:,i,:],axis=0)
    ax.plot(x, mean_ssp, color=colors_scenarios[i], label=scenarios[i])
    plt.fill_between(x, min_ssp, max_ssp, linewidth=0,color=colors_scenarios[i], alpha=alpha_value)

ax.set_xticks(plot_years,[get_years[i] for i in plot_years])
ax.set_xlabel("Year A.D.")
if historical_data:
    ax.set_ylabel(f"$\Delta$ Temperature [K]")
else:
    ax.set_ylabel(var)
ax.legend()
plt.margins(x=0)
fig.savefig(f'{plot_save_dir}/scenario_model_variance_compute_min_max_delta_historical_{historical_data}.{format}')