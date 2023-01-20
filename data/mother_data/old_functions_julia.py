# TODO add option to use different kinds of aggregations
# TODO this function can be used in the res preprocesser as well
# -> find an elegant, generalized solution for that
# ATTENTION right now this only works for mapping from X km to Y km,
     # where Y is 2-fold number of X! (2-times, 4-times, etc.)
def spat_aggregate_single_var(
    self,
    old_res: str = "25_km",
    new_res: str = "50_km",
    scenario: str = "historical",
    var: str = "BC_em_biomassburning", # TODO make this a list option
    overwrite: bool = False,
):
    """ Spatial aggregation of higher res openburning files to the standard
    nominal resolution used for the other files. This function can be used
    multiple times for different desired resolutions. This function is
    intended to use to spatiall aggregate historical openburning files.
    The function can be used equally for other scenarios if necessary
    (see Args). The desired nominal resolution must be given as argument
    (Number_unit: [50_km]).

    Args:
        old_res (str): Resolution the files currently have. E.g. "25 km".
        new_res (str): Resolution the files should have. E.g. "50 km"
        scenario (str): A subdir such as "historical" that describes a
            scenario run by the climate models.
        var (str): Which variable is considered within the scenario. E.g.
            "BC_em_biomassburning".
        overwrite (bool): Indicating if the files should be overwritten if
            they already exist for this resolution.
    """
    # TODO rename longitude and latitude -> must be consistent for all files

    # testing here how to aggregate a normal file, this is a 50km one
    scenario = "historical"
    a_var = "BC_em_anthro"
    abbr_var = "BC" # used instead of "BC_em_biomassburning" in the files!
    a_path = self.raw_path / scenario / a_var / "50_km" / "mon" / "1750"
    a_file = a_path / "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc"
    b_path = self.raw_path / scenario / var / "25_km" / "mon" / "1750"
    b_file = b_path / "input4mips_historical_BC_em_biomassburning_25_km_mon_gn_1750.nc"
    new_file_path = self.raw_path / "None" / "test_file.nc"

    # resolution ratio (old / new res)
    res_ratio = int(old_res.split('_')[0]) / int(new_res.split('_')[0])
    # TODO!!! Desired degree resolution
    res_degree = 0.5
    # aggregation size
    aggr_size = res_ratio**(-1)

    # open both files
    high_res_file = xr.open_dataset(b_file)
    full_original = xr.open_dataset(a_file)

    # sectors
    old_sectors = full_original.sizes["sector"]
    # TODO make this a user param
    aggr_sectors = True # True (default): summarize all sectors to one False: leave them as it is
    new_sectors = 1 if aggr_sectors else old_sectors # we only have 1 sector for biomassburning

    if (not aggr_sectors) and (not "sector" in high_res_file):
        raise ValueError("If you do not want to aggregate sectors, sectors must exists in the high res file! Consider setting aggr_sectors=True.")

    ### create new nc file with lower res for lon and lat ###
    # copy the original dataset (desired dimensions etc)
    if new_sectors < old_sectors: # change the sector dimension if necessary
        copy_original = full_original.where(full_original.sector < new_sectors).dropna(dim="sector")
    elif new_sectors > old_sectors:
        raise ValueError("Trying to create more sectors than available in original file. We are not able to do this.")
    else:
        copy_original = full_original

    # replace with nans
    copy_original[a_var][:, :, :, :] = np.nan

    # rename GHG variable (target var that changes resolution!) if needed
    if var != a_var:
        copy_original[var] = copy_original[a_var]
        copy_original = copy_original.drop(a_var)

    #############################################

    # open both nc files
    a = full_original # TO DEL
    b = copy_original # TO DEL
    #xr.open_dataset(b_file) # TODO: grap first available file here
    #print(a["BC_em_anthro"][11, :, 359, 719]) # month (12), sector (8), lat (360), lon (720)
    #print(b["BC"][11, 719, 1439]) # time (12), lon (720), lat (1440)
    #print(b["BC"][6, 200:500, 1000:1200].values) # time (12), lon (720), lat (1440)

    aggr_file = copy_original

    # replace nans with zeros
    high_res_file = high_res_file.where(~np.isnan(high_res_file[abbr_var][:, :, :]), 0) # later: could be accelerated

    # TODO make this is a looot faster ...
        # calculate lon lat block once separately
        # find a way to use an "apply" method that is faster


    # move over high res file, aggregate and fill the new low res file
    for i_lon, lon in tqdm(enumerate(aggr_file.lon.values), total=aggr_file.lon.size):
        mid_lon = (aggr_size * i_lon) + (0.5 * aggr_size)
        str_lon = int(mid_lon - (0.5 * aggr_size))
        end_lon = int(mid_lon + (0.5 * aggr_size))
        for i_lat, lat in enumerate(aggr_file.lat.values):
            mid_lat = (aggr_size * i_lat) + (0.5 * aggr_size)
            str_lat = int(mid_lat - (0.5 * aggr_size))
            end_lat = int(mid_lat + (0.5 * aggr_size))
            for i_t, t in enumerate(aggr_file.time.values): # maybe use without values here
                if not aggr_sectors:
                    # another for loop over sectors
                    for i_s, s in enumerate(aggr_file.sector.values):
                        high_res_indices = dict(time=i_t,
                                                sector=i_s,
                                                latitude=slice(str_lat, end_lat),
                                                longitude=slice(str_lon, end_lon))
                        high_res_values = high_res_file[abbr_var][high_res_indices]
                        aggr_value = high_res_values.sum().values
                        low_res_coord_labels = dict(time=t, lat=lat, lon=lon, sector=s)
                        aggr_file[var].loc[low_res_coord_labels] = aggr_value
                else:
                    # Attention: problems can arise when names are not "longitude" & "latitude"
                    high_res_indices = dict(time=i_t,
                                            latitude=slice(str_lat, end_lat),
                                            longitude=slice(str_lon, end_lon))
                    high_res_values = high_res_file[abbr_var][high_res_indices]
                    aggr_value = high_res_values.sum().values
                    # check if coordinate sector exists
                    if "sector" in aggr_file.coords:
                        low_res_coord_labels = dict(time=t, lat=lat, lon=lon, sector=0)
                    else:
                        low_res_coord_labels = dict(time=t, lat=lat, lon=lon)
                    aggr_file[var].loc[low_res_coord_labels] = aggr_value
                    #print(aggr_file[var].loc[low_res_coord_labels])

    print("Finished!")
    print(aggr_file)

    # save as new nc file
    aggr_file.to_netcdf(new_file_path)
    exit(0)
    curr_path = self.raw_path / scenario / var

    # check if old res exists
    print(curr_path / old_res)

    # check if new res exists (if overwrite false, exit the function)

    # create folder if new res does not exist yet

    # iterate through subdirs and files

        # create the same subdirs (if they dont exist yet)

        # aggregate the single file (from old dir) to desired res
            # ... do stuff ...

        # save as new file in the new dir

    # finished
    pass
