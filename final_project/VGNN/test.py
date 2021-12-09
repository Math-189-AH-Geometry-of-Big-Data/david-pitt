import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import Counter
import pickle
from tqdm import tqdm
from datetime import datetime
from model import VariationalGNN
from utils import train, evaluate, EHRData, collate_fn
import os
import logging
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

def main():
    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--result_path', type=str, default='.', help='output path of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./mimic', help='input path of processed dataset')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_of_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_of_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--reg', type=str, default="True", help='regularization')
    parser.add_argument('--lbd', type=int, default=1.0, help='regularization')
    parser.add_argument('--state_dict',type=str, default='')

    args = parser.parse_args()
    result_path = args.result_path
    data_path = args.data_path
    in_feature = args.embedding_size
    out_feature =args.embedding_size
    n_layers = args.num_of_layers - 1
    lr = args.lr
    args.reg = (args.reg == "True")
    n_heads = args.num_of_heads
    dropout = args.dropout
    state_dict = args.state_dict
    alpha = 0.1
    BATCH_SIZE = args.batch_size
    number_of_epochs = 1
    eval_freq = 1000

    # Load data
    test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))

    # Create result root
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    result_root = '%s/lr_%s-input_%s-output_%s-dropout_%s'%(result_path, lr, in_feature, out_feature, dropout)
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("Time:%s" %(s))

    # initialize models
    num_of_nodes = test_x.shape[1] + 1
    device_ids = range(torch.cuda.device_count())
    # eICU has 1 feature on previous readmission that we didn't include in the graph

    ############################
    ####### DAVID EDIT!! #######
    ############################

    #torch.cuda.empty_cache() # This may break the model! Fix HERE

    ############################
    ####### DAVID EDIT!! #######
    ############################

    model = VariationalGNN(in_feature, out_feature, num_of_nodes, n_heads, n_layers,
                           dropout=dropout, alpha=alpha, variational=args.reg, none_graph_features=0).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
   
    # Load the state dict

    if state_dict != '':
            model.load_state_dict(torch.load('./{}'.format(state_dict)))

    # Test model
    test_loader = DataLoader(dataset=EHRData(test_x, test_y), batch_size=BATCH_SIZE,
                                collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
    #t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
    print('Evaluating on full test set...')
    
    model.eval()
    loss, _ = evaluate(model, test_loader,len(test_y))
    print(f'AUPRC: {loss}')


if __name__ == '__main__':
    main()
