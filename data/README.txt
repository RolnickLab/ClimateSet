This readme is for internal use, explaining how to setup the code on the cluster.

##Mila Cluster:
ATTENTION: If you want access to the data (without needing to download it),
ask julia.kaltenborn@mila.quebec for access to her climate data directory. Climate data
might be moved to more accessible location later on.

1. Sharing data directories

If you want to access the same data dirs with your co-workers:

1. 1. Make subdirs and dirs searchable -> important: your $USER dir must become searchable [x]

setfacl -m   user:$USER2:x   /network/scratch/$USER/$DIR/$DIR2/
setfacl -m   user:$USER2:x   /network/scratch/$USER/$DIR/
setfacl -m   user:$USER2:x   /network/scratch/$USER/

[x]: exectuable
[w]: writeable
[r]: readable

See also: https://github.com/mila-iqia/mila-docs/pull/121

1. 2. use chmod to give user access to dirs and files within your user dir


##How to use the code:
python -m mother_data.birth

##Explanation of structure:

TODO

### How the downloaded data is structured
Hierarchy:

-- Model
-- Assemble Member
-- Experiment (SSP, Historical)
-- Variable
-- Resolution
-- Final-Files (internally: x_array, severals years, but chunked up in years internally)

Filename:
model_assemble_experiment_var_res_'yearStart'_'yearEnd'.nc

##Open Question:
- BC and BC_antro: BC covers only historical data. Do we need to add past antropogenic BC on top of that?
- BC_antro: exists also for historical values!
- regarding the future BC values: Do we need to add the baseline BC to BC_antro for future scenarios?
- Do we need to add natural GHG sources?? (volcanic? open burning is avail, but also other sources?)

Sources:
https://core.ac.uk/download/pdf/293753522.pdf

Internal Todos:
# TODO: in data_description: which models / scenarios /variables exist
