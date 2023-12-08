n_ep = 200

vocab_size = 5                   # max number of values to choose from for a single parameter, used in the embedding layer
number_features_per_block = 6    # number of features per block
number_blocks = 2                # number of blocks 

conv_n_out_channels = [1, 4, 8, 16, 32]
conv_kernel_size = [3, 5, 7, 15]
pool_type = ['none', 'max', 'avg']
pool_kernel_size = [3, 5, 7]
norm_type = ['none', 'batch', 'instance']
act_funcs = ['none', 'sigmoid', 'tanh', 'relu', 'elu']
