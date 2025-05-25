import os
import random
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from utils.utils import fix_seed

# ---
# parameters
# ---

load_path = 'dataset/weather_data/'  # path to the dataset root folder
save_path = 'dataset/gen_anomalies/' # path to save the generated anomalies

# CHANGED: Only process the file you have
dataset_names = ["IHAMPS1"]  # Changed from the full list
feat_to_keep = ["Temperature_C", "Humidity_%", "Pressure_hPa"]

contamination_rate = 0.2 
train_test_split_ratio = 0.6

seed = 19
fix_seed(seed)

# Create all necessary directories
os.makedirs(save_path, exist_ok=True)
os.makedirs('dataset/synth_ts_data', exist_ok=True)  # Added this line


def load_data(path, filename):
    csv_load_path = os.path.join(path, filename)
    return pd.read_csv(csv_load_path, index_col=0)

def save_data(df, path, filename):
    csv_save_path = os.path.join(path, filename)
    df.to_csv(csv_save_path)

def triOut_remove(reg_data, model='GMM', rate=10): 
    X_trn = reg_data.iloc[:, :]
    if model == 'GMM':
        fit_model = GaussianMixture(n_components=3, n_init=10, reg_covar=1e-3, random_state=seed)
    scaler = StandardScaler()  # Removed random_state parameter
    X_trn = scaler.fit_transform(X_trn)
    fit_model.fit(X_trn)
    densities = fit_model.score_samples(X_trn)
    density_threshold = np.percentile(densities, rate)
    purified_data = reg_data.loc[densities >= density_threshold].reset_index(drop=True)
    
    return purified_data

def local_generation(reg_data, n_insts=10000):
    alpha = 5
    reg_inst = reg_data.iloc[:,:].values
    data_trn, data_tst = train_test_split(reg_inst, test_size=0.3, random_state=0)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 3)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type, random_state=seed
            )
            gmm.fit(data_trn)
            bic.append(gmm.bic(data_tst))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    # determine local outlier distribution model
    local_gmm = sklearn.base.clone(best_gmm)
    local_gmm.weights_ = best_gmm.weights_
    local_gmm.means_ = best_gmm.means_
    local_gmm.covariances_ = alpha*best_gmm.covariances_ # stretching distribution space
    
    # Sample local outliers from outlier distribution
    local_insts = local_gmm.sample(n_insts) # generated instances
    local_insts = pd.DataFrame(local_insts[0], columns=reg_data.columns)
    local_insts['label'] = np.tile(1, local_insts.shape[0]) # Add label of outliers as 1

    # Return local outliers in form of data frame
    return local_insts


def inst_filter(reg_data, out_data, classifier='KNN', n_insts=10000, maxRem=50):
    removed = np.inf
    reg_inst = reg_data.copy()
    reg_inst["label"] = np.tile(0, reg_inst.shape[0])
    out_source = out_data.copy()
    idx = random.sample(out_source.index.to_list(), n_insts)
    out_inst = out_source.loc[idx]
    out_source.drop(idx, axis=0)
    if classifier=='KNN':
        clf = KNeighborsClassifier()
    while removed > maxRem:  
        data_trn = pd.concat((reg_inst, out_inst), axis=0).reset_index(drop=True)
        X, y = data_trn.iloc[:, :-1].values.astype('float32'), data_trn.iloc[:, -1].values.astype('int32')
        clf.fit(X, y)
        y_out = clf.predict(out_inst.iloc[:, :-1].values)
        out_inst_old = out_inst.loc[y_out==1]
        removed = np.sum(np.where(y_out==0, 1, 0))
        print(removed)
        idx = random.sample(out_source.index.to_list(), removed)
        out_inst_new = out_source.loc[idx]
        out_source.drop(idx, axis=0)
        out_inst = pd.concat((out_inst_old, out_inst_new), axis=0).reset_index(drop=True)
        
    return out_inst


def global_generation(reg_data, n_insts=10000):
    # reg_data = reg_data.drop('label', axis=1)
    info = reg_data.describe()
    attri_mins = 0.9*info.loc['min'].values
    attri_maxs = 1.1*info.loc['max'].values
    attri_scales = attri_maxs - attri_mins
    rv = scipy.stats.uniform(loc=[attri_mins], scale=[attri_scales])
    global_insts = []

    # create 1000 global outliers
    for i in range(n_insts):
        sample = rv.rvs(size=attri_mins.shape)
        global_insts.append(sample)

    global_insts = pd.DataFrame(global_insts, columns=reg_data.columns)
    global_insts['label'] = np.tile(1, global_insts.shape[0])
    
    return global_insts


def generate_anomalies(load_path, save_path, dataset_name, feat_to_keep):
    reg_data = load_data(load_path, f'{dataset_name}.csv')
    reg_data = reg_data[feat_to_keep]
    reg_data.dropna(inplace=True)
    reg_data = reg_data.reset_index(drop=True)

    # local outliers
    local_insts = local_generation(reg_data, n_insts=50000)
    local_outs = inst_filter(reg_data, local_insts, n_insts=15000)

    # global outliers
    global_insts = global_generation(reg_data, n_insts=50000)
    global_outs = inst_filter(reg_data, global_insts, n_insts=15000)

    # save files
    save_data(local_outs, save_path, f'locOuts_{dataset_name}.csv')
    save_data(global_outs, save_path, f'gloOuts_{dataset_name}.csv')


# gen anom
def get_anomalies(dataset_name, feature_name):
    loc_anom_path = f"dataset/gen_anomalies/locOuts_{dataset_name}.csv"
    loc_anom_df = pd.read_csv(loc_anom_path)
    loc_anom_df = loc_anom_df.dropna()
    loc_anom_feat = loc_anom_df[feature_name]

    glob_anom_path = f"dataset/gen_anomalies/gloOuts_{dataset_name}.csv"
    glob_anom_df = pd.read_csv(glob_anom_path)
    glob_anom_df = glob_anom_df.dropna()
    glob_anom_feat = glob_anom_df[feature_name]

    anom_feat = pd.concat([loc_anom_feat, glob_anom_feat], axis=0)
    return anom_feat


def contam_save_data(dataset, dataset_name, feature_name):
    feat = dataset[feature_name]
    feat = feat.dropna()
    feat = feat.values

    window_size = 100
    step = 50

    lb = 0
    hb = window_size
    train_size = len(feat)/window_size*window_size/step*train_test_split_ratio

    train_out = []
    train_gt = []
    test_out = []
    test_gt = []

    anom_feat = get_anomalies(dataset_name, feature_name)

    i = 0
    while hb<len(feat):
        normal_sample = feat[lb:hb]
        
        # train data
        if i<train_size:
            train_out.append(normal_sample)
            train_gt.append(0)

        # contaminated test data
        else:
            normal_sample = feat[lb:hb]
            r = random.random()
            if r<contamination_rate:
                ts = normal_sample.copy()
                anom_t = random.randint(int(0.25*window_size), int(0.75*window_size))

                id = random.randint(0, len(anom_feat)-1)
                ts[anom_t] = anom_feat.values[id]
                while np.abs(ts.mean()-anom_feat.values[id])>50: # not need to try to detect point that are obviously anomalous (global outliers that are too far from the normal data distribution)
                    ts = normal_sample.copy()
                    id = random.randint(0, len(anom_feat)-1)
                    ts[anom_t] = anom_feat.values[id]

                test_out.append(ts)
                test_gt.append(1)
                
            else:
                test_out.append(normal_sample)
                test_gt.append(0)

        lb += step 
        hb += step 
        i += 1

    train_out_df = pd.DataFrame(train_out)
    train_out_df["gt"] = train_gt
    train_out_df.to_csv(f"dataset/synth_ts_data/train_{dataset_name}_{feature_name}.csv", index=False)

    test_out_df = pd.DataFrame(test_out)
    test_out_df["gt"] = test_gt
    test_out_df.to_csv(f"dataset/synth_ts_data/test_{dataset_name}_{feature_name}.csv", index=False)


if __name__ == "__main__":
    
    print("Starting data preparation...")
    
    # Check if files exist before processing
    for dataset_name in dataset_names:
        data_path = os.path.join(load_path, f'{dataset_name}.csv')
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping...")
            continue
            
        print(f"\nProcessing {dataset_name}...")
        dataset = pd.read_csv(data_path)
        
        # Check which features are available
        available_features = [f for f in feat_to_keep if f in dataset.columns]
        missing_features = [f for f in feat_to_keep if f not in dataset.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features} in {dataset_name}")
        
        if not available_features:
            print(f"Error: No required features found in {dataset_name}")
            continue
            
        print(f"Generating anomalies for features: {available_features}")
        generate_anomalies(load_path, save_path, dataset_name, available_features)

        for feature_name in available_features:
            print(f"Creating contaminated datasets for {feature_name}...")
            contam_save_data(dataset, dataset_name, feature_name)
    
    print("\nData preparation completed!")
    print(f"Generated files in:")
    print(f"  - {save_path} (anomaly files)")
    print(f"  - dataset/synth_ts_data/ (train/test files)")
