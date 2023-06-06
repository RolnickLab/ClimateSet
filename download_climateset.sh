# Change the directory where the data will be downloaded below
data_dir="Climateset_DATA"
mkdir -p ${data_dir}/inputs
mkdir -p ${data_dir}/outputs

# Download the entire dataset: (~around 64GB)
# Uncomment the line and comment the for loop for downloading only NorESM2-LM


echo "Downloading input4mips data..."
curl https://object-arbutus.cloud.computecanada.ca/causalpaca/inputs/input4mips.tar.gz --output ${data_dir}/inputs/input4mips.tar.gz
echo "Done."

echo "Downloading compressed CMIP6 models data..."
for x in AWI-CM-1-1-MR.tar.gz BCC-CSM2-MR.tar.gz CAMS-CSM1-0.tar.gz CAS-ESM2-0.tar.gz CESM2.tar.gz CESM2-WACCM.tar.gz CMCC-CM2-SR5.tar.gz CMCC-ESM2.tar.gz CNRM-CM6-1-HR.tar.gz EC-Earth3.tar.gz EC-Earth3-Veg-LR.tar.gz EC-Earth3-Veg.tar.gz FGOALS-f3-L.tar.gz GFDL-ESM4.tar.gz INM-CM4-8.tar.gz INM-CM5-0.tar.gz MPI-ESM1-2-HR.tar.gz MRI-ESM2-0.tar.gz NorESM2-LM.tar.gz NorESM2-MM.tar.gz TaiESM1.tar.gz ;do  curl https://object-arbutus.cloud.computecanada.ca/causalpaca/outputs/$x --output ${data_dir}/outputs/$x; done

#curl https://object-arbutus.cloud.computecanada.ca/causalpaca/outputs/NorESM2-LM.tar.gz --output ${data_dir}/outputs/NorESM2-LM.tar.gz

echo "Done. Finished downloading the compressed files, now extracting and deleting compressed files!"

cd ${data_dir}/inputs
for x in `ls |grep .gz`;
do tar -xzf $x;
do rm -f $x;
done;

cd ../../${data_dir}/outputs
for x in `ls |grep .gz`;
do tar -xzf $x;
do rm -f $x;
done;

echo "Done. Finished downloading Climateset data! :)"
