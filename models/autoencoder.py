# ---
# imports
# ---
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from datetime import timedelta
from tqdm import tqdm 
import time

from utils.data_loaders import get_data_loaders
from utils.utils import translate_sequence
from config.search_space import *
from config.autoencoder_params import *


# ---
# autoencoder model definition
# ---
class EncoderBlock(nn.Module):
    def __init__(self, enc_conv_layer, enc_pool_layer=None, enc_norm_layer=None, enc_act_func=None):
        super(EncoderBlock, self).__init__()
        self.encoding_block = nn.Sequential()
        self.encoding_block.add_module('conv', enc_conv_layer)
        if enc_pool_layer!=None: self.encoding_block.add_module('pool', enc_pool_layer)
        if enc_norm_layer!= None: self.encoding_block.add_module('norm', enc_norm_layer)
        if enc_act_func!= None: self.encoding_block.add_module('act', enc_act_func)

    def forward(self, x):
        return self.encoding_block(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, dec_tconv_layer, dec_norm_layer=None, dec_act_func=None):
        super(DecoderBlock, self).__init__()
        self.decoding_block = nn.Sequential()
        self.decoding_block.add_module('tconv', dec_tconv_layer)
        if dec_norm_layer!=None: self.decoding_block.add_module('norm', dec_norm_layer)
        if dec_act_func!= None: self.decoding_block.add_module('act', dec_act_func)
        
    def forward(self, x):
        return self.decoding_block(x)
    

# ---
# build autoencoder from param sequence
# ---
class AE(nn.Module):
    """Autoencoder model definition, allows building the architecture from a sequence of parameters."""

    def __init__(self, sequence):
        super(AE, self).__init__()
        encoder_blocks_list, decoder_blocks_list = self.build_model(sequence)
        self.encoder = nn.ModuleList(encoder_blocks_list)
        self.decoder = nn.ModuleList(decoder_blocks_list)

    def build_model(self, sequence):
        # builds model architecture from token sequence
        encoder_blocks_list = []
        decoder_blocks_list = []
        tr_seq = translate_sequence(sequence)
        conv_n_in_channels = 1
        prev_enc_out_width = 100

        for i, block in enumerate(tr_seq):
            conv_n_out_channels, conv_kernel_size, pool_type, pool_kernel_size, norm_type, act_func = block.values()
            # print(block.values())

            # Encoder
            # params
            conv_pad = 1
            conv_stride = 1
            pool_stride = 2
            conv_out_width = (prev_enc_out_width + 2*conv_pad - conv_kernel_size)//conv_stride + 1
            pool_out_width = (conv_out_width-pool_kernel_size)//pool_stride + 1 if pool_type in ['avg', 'max'] else conv_out_width
            
            # layers choices
            enc_conv_layer = nn.Conv1d(conv_n_in_channels, conv_n_out_channels, conv_kernel_size, conv_stride, conv_pad)

            if pool_type=='avg': enc_pool_layer = nn.AvgPool1d(pool_kernel_size, pool_stride)
            elif pool_type=='max': enc_pool_layer = nn.MaxPool1d(pool_kernel_size, pool_stride)
            else: enc_pool_layer = None

            if norm_type=='batch': enc_norm_layer = nn.BatchNorm1d(conv_n_out_channels)
            elif norm_type=='instance': enc_norm_layer = nn.InstanceNorm1d(conv_n_out_channels)
            else: enc_norm_layer=None

            if act_func=='relu': enc_act_func = nn.ReLU()
            elif act_func=='elu': enc_act_func = nn.ELU()
            elif act_func=='sigmoid': enc_act_func = nn.Sigmoid()
            elif act_func=='tanh': enc_act_func = nn.Tanh()
            else: enc_act_func = None

            encoder_blocks_list.append(
                EncoderBlock(
                    enc_conv_layer,
                    enc_pool_layer,
                    enc_norm_layer,
                    enc_act_func
                )
            )

            # Decoder
            # params
            convT_stride = 1
            convT_in_width = pool_out_width
            convT_out_width = prev_enc_out_width
            convT_kernel_size = int(convT_out_width-(convT_in_width-1)*convT_stride)

            dec_tconv_layer = nn.ConvTranspose1d(conv_n_out_channels, conv_n_in_channels, kernel_size=convT_kernel_size, stride=convT_stride)
            
            # layers choices
            if norm_type=='batch': dec_norm_layer = nn.BatchNorm1d(conv_n_in_channels)
            elif norm_type=='instance': dec_norm_layer = nn.InstanceNorm1d(conv_n_in_channels)
            else: dec_norm_layer=None

            if act_func=='relu': dec_act_func = nn.ReLU()
            elif act_func=='elu': dec_act_func = nn.ELU()
            elif act_func=='sigmoid': dec_act_func = nn.Sigmoid()
            elif act_func=='tanh': dec_act_func = nn.Tanh()
            else: dec_act_func = None

            decoder_blocks_list.insert(
                0,
                DecoderBlock(
                    dec_tconv_layer,
                    dec_norm_layer,
                    dec_act_func
                )
            )

            conv_n_in_channels = conv_n_out_channels
            prev_enc_out_width = pool_out_width

        return encoder_blocks_list, decoder_blocks_list

    def forward(self, x):
        encode = x
        for encoder_block in self.encoder:
            encode = encoder_block(encode)
        decode = encode
        for decoder_block in self.decoder:
            decode = decoder_block(decode)
        return decode
    

# ---
# train and test architecture
# ---
class Architecture():
    """Autoencoder model training and testing."""

    def __init__(self, sequence, train_dataset_path, test_dataset_path):
        self.model = AE(sequence)
        self.model= nn.DataParallel(self.model) # for multi-gpu training
        self.threshold = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader, self.len_train_set = get_data_loaders(train_dataset_path,test_dataset_path)

    def train(self):
        """Trains the autoencoder model."""
        self.metrics = defaultdict(list)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=w_d)
        
        self.model.train()
        # start = time.time()
        for epoch in range(epochs):
            # ep_start = time.time()
            running_loss = 0.0
            for i, (data, _) in tqdm(enumerate(self.train_loader), disable=disable_tqdm):
                original_ts = data.to(self.device)
                reconstructed_ts = self.model(original_ts)
                loss = self.criterion(original_ts, reconstructed_ts)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss/self.len_train_set
            self.metrics['train_loss'].append(epoch_loss)
            # ep_end = time.time()
            # print('-----------------------------------------------')
            # print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1,epochs,epoch_loss))
            # print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
        # end = time.time()
        # print('-----------------------------------------------')
        # print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

    def choose_anomaly_threshold(self):
        """Chooses the anomaly threshold based on the reconstruction error distribution."""
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (data_batch, _) in tqdm(enumerate(self.train_loader), disable=disable_tqdm):
                for sequence in data_batch:
                    sequence = torch.unsqueeze(sequence, 0)
                    original_ts = sequence.to(self.device)
                    reconstructed_ts = self.model(original_ts)
                    # loss = self.criterion(original_ts, reconstructed_ts)
                    loss = torch.max(torch.abs(original_ts - reconstructed_ts))
                    losses.append(loss.item())

        loss_threshold = np.percentile(losses, q=99)
        self.threshold = loss_threshold

    def fitness(self, data_type="val", return_metrics=False):
        """Calculates the fitness of the sampled architecture."""
        self.choose_anomaly_threshold()
        threshold = self.threshold

        if data_type=="val":
            data_loader = self.val_loader
        elif data_type=="test":
            data_loader = self.test_loader

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        self.model.eval()
        with torch.no_grad():
            for i, (data, gt) in tqdm(enumerate(data_loader), disable=disable_tqdm):
                original_ts = data.to(self.device)
                reconstructed_ts = self.model(original_ts)
                # loss = self.criterion(original_ts, reconstructed_ts)
                loss = torch.max(torch.abs(original_ts - reconstructed_ts))
                if loss >= threshold:
                    if gt == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if gt == 1:
                        fn += 1
                    else:
                        tn += 1
    
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        # print('[TP] {}\t[FP] {}'.format(tp, fp))
        # print('[FN] {}\t[TN] {}'.format(fn, tn))
        # print(f"F1 score = {f1}")
        if return_metrics:
            return f1, precision, recall
        return f1
    