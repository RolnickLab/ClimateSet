

# Supported Model sources

MODEL_SOURCES = {
    "NorESM2-LM" : {
    "url": "https://dap.ceda.ac.uk/thredds/catalog/badc/cmip6/data/CMIP6/catalog.xml",
    "center": "NCC"
    }
}


VAR_SOURCE_LOOKUP = {
    'model' : ['tas', 'pr'],
     'raw' : []
     }


# filepath to var to res Mapping
VAR_RES_MAPPING_PATH="/home/charlie/Documents/MILA/causalpaca/data/data_description/mappings/variableid2tableid.csv"



GRIDDING_HIERACHY = ['gn']

# skip subhr because only diagnostics for specific places
REMOVE_RESOLUTONS = ['suhbr'] # resolution endings to remove e.g. kick CFsubhr if this contains 'subhr'
