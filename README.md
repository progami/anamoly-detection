## RLNAS for Autoencoder based Anomaly Detection in Time Series Data

This repository contains the official implementation of our research paper *"Neural Architecture Search for Anomaly Detection in Time Series Data of Smart Buildings: A Reinforcement Learning Approach for Optimal Autoencoder Design"*

### Setup
``````
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
``````

### Running
1. Generate the dataset: 
    
    Running `prepare_data.py` will generate contextual and point anomalies (using the Gaussian Mixture Model and Multivariate Uniform Distribution methods, as described in the paper) and inject them into clean data. 
    
    Path and contamination parameters need to be set first. 
    
    Generated train and test data are saved to `dataset/`.

2. Set the Neural Architecture Search parameters:
    - `config/autoencoder_params.py` contains training parameters of the Autoencoder architecture. (not included in the search) 
    - `config/controller_params.py` contains the parameters of the Controller network.
    - `config/search_space.py` defines the search space, specifying the size of the target Autoencoder, the possible parameters, and the number of episodes to run the NAS process.

3. Run the NAS framework: `python main.py` with the arguments `--train_dataset_path`, `--test_dataset_path` and `--output_file`:

    This will execute the RLNAS framework on the training data and subsequently evaluate the best discovered architecture on the test data. Performance metric and top architectures will be saved to the file specified by the `--output_file` argument.

### Comments 
- For collecting the weather dataset, we use code from the [the-weather-scraper](https://github.com/Karlheinzniebuhr/the-weather-scraper) project.

### Citation
TBA
