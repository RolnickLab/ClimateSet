# ***ClimateSet*** - : A Large-Scale Climate Model Dataset for Machine Learning

## Official implementation for the data downloader & processor

Abstract: *Climate models have been key for assessing the impact of climate change and simulating future climate scenarios depending on humanity’s socioeconomic choices.

The machine learning (ML) community has taken an increased interest in supporting climate scientists’ efforts on various tasks such as climate emulation, downscaling, and prediction tasks. Many of those tasks have been addressed on datasets created with single climate models. However, both the climate science and the ML communities have communicated that to address those tasks at scale, we need large, consistent, and ML-ready climate model datasets. Here, we introduce ClimateSet, a dataset containing the inputs and outputs of 36 climate models from the CMIP6 and Input4MIPs archives. In addition, we provide a modular dataset pipeline for retrieving and pre-processing additional climate models and scenarios. 

We showcase the potential of our dataset by using it as a benchmark for ML-based climate emulation. We gain new insights about the performance and generalization capabilities of the different ML models by analyzing them across different climate models. Furthermore, the dataset is used to train an ML model on all 36 climate models, i.e. not only one specific climate model but the entire CMIP6 archive is emulated. With this, we can quickly project new climate scenarios capturing the inter-model variability of climate models - similar to the “averaged climate scenarios” provided to policymakers. We believe ClimateSet will create the basis needed for the ML community to tackle climate model related tasks at scale.*

This repositorcy contains 2 independent pathways: Databuilding and emulation.

### Data Building
If you wish to create an individual climate dataset or extend the core dataset provided by ClimateSet please refer to [downloader](README_downloader.md) and [preprocessor](README_preprocessor.md) pages for further information.

### Emulation
if you wish to set up your own experiments, reproduce our benchmarks on the core dateset or your individual dataset, please refer to the [emulator](README_emulator.md) page for further information.

## Development

This repository is currently under active development and you may encounter bugs with some functionality. 
Any feedback, extensions & suggestions are welcome!
