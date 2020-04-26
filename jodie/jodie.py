'''
Train the JODIE model on given dataset for dropout prediction:
'''
import os
import csv
import sys
import math
import random

import subprocess
import numpy as np
from collections import namedtuple
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def create_folders(): # done
    commands = '''mkdir -p data/
                  mkdir -p results/
                  mkdir -p saved_models/
                  mkdir -p saved_models/reddit/
                  mkdir -p saved_models/wikipedia/
                  mkdir -p saved_models/mooc/
                  mkdir -p saved_models/lastfm/'''

    for command in commands.split("\n"):
        os.system(command.strip())


def download_datasets(): # done
    files = os.listdir("data")

    commands = '''wget http://snap.stanford.edu/jodie/mooc.csv -P data/'''

                # '''wget http://snap.stanford.edu/jodie/reddit.csv -P data/
                #   wget http://snap.stanford.edu/jodie/wikipedia.csv -P data/
                #   wget http://snap.stanford.edu/jodie/mooc.csv -P data/
                #   wget http://snap.stanford.edu/jodie/lastfm.csv -P data/'''

    for command in commands.split('\n'): # bit ugly dataset name extractor
        f = command[command.find("jodie")+6:command.find(".csv")+4]
        if f not in files: subprocess.run(command.strip().split())


def load_dataset(datapath, time_scaling=True):
    '''
    load rows of: user, item, timestamp(cardinal), state_label, array_of_features
    
    state label = 1 whenever the user state changes, 0 otherwise.
    if no state labels, use 0 as placeholder.

    feature list can be as long as desired, should be at least 1-D
    if no features, use 0 as placeholder.
    '''

    user_sequence = []
    item_sequence = []
    timestamp_sequence = []
    label_sequence = []
    feature_sequence = []
    
    print("\n\n**** Loading %s ****" % datapath)
    f = open(datapath,"r")
    f.readline() # header
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        timestamp_sequence.append(float(ls[2])) 
        label_sequence.append(int(ls[3])) # 1 for state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)
    timestamp_sequence -= timestamp_sequence[0]

    # print "Formating item sequence" 
    nodeid = 0; item2id = {} # indexing, manually perform DB's work
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float) # just to use default 0.0
    for timestamp, item in zip(timestamp_sequence, item_sequence):
        if item not in item2id:
            item2id[item] = nodeid; nodeid += 1
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence] # ???usage?

    # print "Formating user sequence"
    nodeid = 0; user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items) # like -1 represents none?
    for user, item, timestamp  in zip(user_sequence, item_sequence, timestamp_sequence):
        if user not in user2id:
            user2id[user] = nodeid; nodeid += 1
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        # print "Scaling timestamps", + laplacian smooth
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")

    # too much returned value... why bother dropping user_seq and item_seq, just return xx_id_seq?
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        label_sequence]


class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class JODIE(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(JODIE,self).__init__()

        # print("*** Initializing the JODIE model ***")
        self.model_name = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        # print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        # print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        # print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        # print("*** JODIE initialization complete ***\n\n")
        
    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out


def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path="./"):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.task)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
    torch.save(state, filename)
    print("*** Saved embeddings and model to file: %s ***\n\n" % filename)


def load_model(model, optimizer, args, epoch, path="./"):
    model_name = args.model
    filename = path + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.task, model_name, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print("Loading saved embeddings and model: %s" % filename)
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).to(dev))
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).to(dev))
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).to(dev))
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).to(dev))
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# INITIALIZE T-BATCH VARIABLES ????
total_reinitialization_count = 0

def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).to(dev)[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss

# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD ???
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()

tbatch_sizes = []

def train(epoch_num, from_epoch=-1):
    TrainArgs = namedtuple('TrainArgs',
                ['task', 'model', 'epochs', 'embedding_dim', 
                'state_change', 'train_proportion', 'datapath'])
    # need train_proportion <= 0.8
    args = TrainArgs('mooc', 'jodie', epoch_num, 128, True, 0.8, 'data/mooc.csv')

    # LOAD DATA
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
    item2id, item_sequence_id, item_timediffs_sequence, 
    timestamp_sequence, feature_sequence, y_true] = load_dataset(args.datapath)

    num_interactions = len(user_sequence_id)
    num_users = len(user2id) 
    num_items = len(item2id) + 1 # one extra item for "none-of-these"
    num_features = len(feature_sequence[0])
    true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 to avoid divide 0 error. 

    stats = (num_users, num_items, num_interactions, sum(y_true), len(y_true))
    print("*** Loaded network statistics: ***")
    print("\n  %d users\n  %d items\n  %d interactions\n  %d/%d state changes ***\n\n" % stats)

    # SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
    train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
    test_start_idx = int(num_interactions * (args.train_proportion+0.1))
    test_end_idx = int(num_interactions * (args.train_proportion+0.2)) # tbatch boundary? where?

    # SET BATCHING TIMESPAN
    '''
    Timespan is the frequency at which the batches are created and the JODIE model is trained. 
    As the data arrives in a temporal order, the interactions within a timespan are added into batches
    using the T-batch algorithm. The batches are then used to train JODIE. 
    Longer timespans mean more interactions are processed and the training time is reduced, 
    however it requires more GPU memory. Longer timespan leads to less frequent model updates. 
    ''' # batching in time dimension is a cool idea.
    tbatch_timespan = (timestamp_sequence[-1] - timestamp_sequence[0]) / 500 # like sliding window for ASR

    # INITIALIZE MODEL AND PARAMETERS
    model = JODIE(args, num_features, num_users, num_items).to(dev)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=torch.Tensor([1,true_labels_ratio])).to(dev)
    MSELoss = nn.MSELoss()

    # INITIALIZE EMBEDDING
    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(dev), dim=0))
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(dev), dim=0))
    model.initial_user_embedding = initial_user_embedding # shouldn't be called model.user_embedding = initial_user_embedding?
    model.initial_item_embedding = initial_item_embedding

    user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
    item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
    item_embedding_static = Variable(torch.eye(num_items).to(dev)) # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).to(dev)) # one-hot vectors for static embeddings 

    # INITIALIZE MODEL
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # if from_epoch >= 0:
        # model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, from_epoch)

    # TRAIN MODEL
    print("*** Training the %s model from epoch %d  to epoch %d***" % (args.model, from_epoch, args.epochs))
    for ep in range(from_epoch+1, args.epochs):

        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(dev))
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(dev))

        optimizer.zero_grad()
        reinitialize_tbatches() #???
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX 
        for j in range(train_end_idx):

            # READ INTERACTION J
            userid = user_sequence_id[j]
            itemid = item_sequence_id[j]
            feature = feature_sequence[j]
            user_timediff = user_timediffs_sequence[j]
            item_timediff = item_timediffs_sequence[j]

            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
            tbatch_to_insert = max(tbatchid_user[userid], tbatchid_item[itemid]) + 1 
            tbatchid_user[userid] = tbatch_to_insert 
            tbatchid_item[itemid] = tbatch_to_insert

            current_tbatches_user[tbatch_to_insert].append(userid)
            current_tbatches_item[tbatch_to_insert].append(itemid)
            current_tbatches_feature[tbatch_to_insert].append(feature)
            current_tbatches_interactionids[tbatch_to_insert].append(j)
            current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
            current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
            current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                # ITERATE OVER ALL T-BATCHES
                for i in range(len(current_tbatches_user)):
                    total_interaction_count += len(current_tbatches_interactionids[i])

                    # LOAD THE CURRENT TBATCH
                    tbatch_userids = torch.LongTensor(current_tbatches_user[i]).to(dev) # Recall "current_tbatches_user[i]" has unique elements
                    tbatch_itemids = torch.LongTensor(current_tbatches_item[i]).to(dev) # Recall "current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = torch.LongTensor(current_tbatches_interactionids[i]).to(dev) 
                    feature_tensor = Variable(torch.Tensor(current_tbatches_feature[i])).to(dev) # Recall "current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(torch.Tensor(current_tbatches_user_timediffs[i])).to(dev).unsqueeze(1)
                    item_timediffs_tensor = Variable(torch.Tensor(current_tbatches_item_timediffs[i])).to(dev).unsqueeze(1)
                    tbatch_itemids_previous = torch.LongTensor(current_tbatches_previous_item[i]).to(dev)
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    # print("tbatch size: " + str(len(tbatch_userids)))
                    tbatch_sizes.append(len(tbatch_userids))

                    user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING                            
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                    item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output  

                    user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                    # print("current loss: " + str(loss))
                    # CALCULATE STATE CHANGE LOSS
                    if args.state_change:
                        loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss) 

                # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # RESET LOSS FOR NEXT T-BATCH
                loss = 0
                item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
                
                # REINITIALIZE
                reinitialize_tbatches()
                tbatch_to_insert = -1

        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % total_loss)
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

    # END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
    print("\n\n*** Training complete. Saving final model. ***\n\n")
    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)


def evaluate_state_change_prediction(epoch_id):
    # INITIALIZE PARAMETERS
    EvalArgs = namedtuple("EvalArgs",
            ['task', 'model', 'epoch', 'embedding_dim',
            'train_proportion', 'state_change', 'datapath'])
    # Training sequence proportion cannot be greater than 0.8
    # No state change prediction for lastfm dataset
    args = EvalArgs('mooc', 'jodie', epoch_id, 128, 0.8, True, 'data/mooc.csv')

    # CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
    output_fname = "results/state_change_prediction_%s_%s.txt" % (args.task, args.model)
    if os.path.exists(output_fname):
        f = open(output_fname, "r")
        search_string = 'Test performance of epoch %d' % args.epoch
        for l in f:
            l = l.strip()
            if search_string in l:
                print("Output file already has results of epoch %d" % args.epoch)
                sys.exit(0)
        f.close()

    # LOAD NETWORK
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
    item2id, item_sequence_id, item_timediffs_sequence, \
    timestamp_sequence, \
    feature_sequence, \
    y_true] = load_dataset(args.datapath)
    num_interactions = len(user_sequence_id)
    num_features = len(feature_sequence[0])
    num_users = len(user2id)
    num_items = len(item2id) + 1
    true_labels_ratio = len(y_true)/(sum(y_true)+1)
    print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))
            
    # SET TRAIN, VALIDATION, AND TEST BOUNDARIES
    train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
    test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
    test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

    # SET BATCHING TIMESPAN
    '''
    Timespan indicates how frequently the model is run and updated. 
    All interactions in one timespan are processed simultaneously. 
    Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
    At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
    '''
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / 500 

    # INITIALIZE MODEL PARAMETERS
    model = JODIE(args, num_features, num_users, num_items).to(dev)
    weight = torch.Tensor([1,true_labels_ratio]).to(dev)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()

    # INITIALIZE MODEL
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    # LOAD THE MODEL
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)
    if train_end_idx != train_end_idx_training:
        sys.exit('Training proportion during training and testing are different. Aborting.')

    # SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

    # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
    item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()

    user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()

    validation_predicted_y = []
    test_predicted_y = []
    validation_true_y = []
    test_true_y = []

    ''' 
    Here we use the trained model to make predictions for the validation and testing interactions.
    The model does a forward pass from the start of validation till the end of testing.
    For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
    This is used to calculate the rank of the true item the user actually interacts with.

    After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
    This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
    Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
    '''
    tbatch_start_time = None
    loss = 0
    # FORWARD PASS
    print("*** Making state change predictions by forward pass (no t-batching) ***")
    for j in range(train_end_idx, test_end_idx):
        if j % 10000 == 0:
            print('%dth interaction for validation and testing' % j)

        # LOAD INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # LOAD USER AND ITEM EMBEDDING
        dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        user_embedding_input = user_embeddings[dtype([userid])]
        user_embedding_static_input = user_embeddings_static[dtype([userid])]
        item_embedding_input = item_embeddings[dtype([itemid])]
        item_embedding_static_input = item_embeddings_static[dtype([itemid])]
        feature_tensor = Variable(torch.Tensor(feature).to(dev)).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).to(dev)).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).to(dev)).unsqueeze(0)
        item_embedding_previous = item_embeddings[dtype([itemid_previous])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[dtype([itemid_previous])], user_embedding_static_input], dim=1)
        
        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

        # CALCULATE STATE CHANGE LOSS
        if args.state_change:
            loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_() 

        # PREDICT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
        prob = model.predict_label(user_embedding_output)

        # ADD PREDICTION TO VALIDATION OR TEST ARRAYS
        if j < test_start_idx:
            validation_predicted_y.extend(prob.data.cpu().numpy())
            validation_true_y.extend([y_true[j]])
        else:
            test_predicted_y.extend(prob.data.cpu().numpy())
            test_true_y.extend([y_true[j]])

    # CALCULATE THE PERFORMANCE METRICS
    validation_predicted_y = np.array(validation_predicted_y)
    test_predicted_y = np.array(test_predicted_y)

    performance_dict = dict()
    auc = roc_auc_score(validation_true_y, validation_predicted_y[:,1])
    performance_dict['validation'] = [auc]

    auc = roc_auc_score(test_true_y, test_predicted_y[:,1])
    performance_dict['test'] = [auc]

    # PRINT AND SAVE THE PERFORMANCE METRICS
    fw = open(output_fname, "a")
    metrics = ['AUC']

    print('\n\n*** Validation performance of epoch %d ***' % args.epoch)
    fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)

    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
        fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")

    print('\n\n*** Test performance of epoch %d ***' % args.epoch)
    fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['test'][i]))
        fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

    fw.flush()
    fw.close()

if __name__ == '__main__':
    create_folders()
    download_datasets()
    epoch_num = 50
    train(epoch_num) # from_epoch=9)
    # for i in range(epoch_num):
    evaluate_state_change_prediction(epoch_num-1)
