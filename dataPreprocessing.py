#%%
import torch
import re
import string 
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#%%
class dataBuilder:
    def __init__(self,project_dir):
        self.project_dir=project_dir
        self.data_dir = project_dir + '/rumor_detection_acl2017'
        self.iterator = 0
        self.samples = []
        self.data = []
        self.train_data = []
        self.test_data = []
        self.train_loader = []
        self.test_loader = []
        self.X_train=[]
        self.X_test=[]
        self.y_train = []
        self.y_test = []
        self.X_train_stack = []
        self.X_test_stack = []
        self.y_train_stack =[]
        self.y_test_stack = []
        self.X_train_np=[]
        self.X_test_np = []
        self.y_train_np=[]
        self.y_test_np=[]
        
    def load(self):
        self.samples = np.load(self.project_dir+'/processedData.npy',allow_pickle=True)
        self.samples = pd.DataFrame(self.samples,columns = ['sample_id','x','y','edge_index'])
        tweet_ids= dict(enumerate(self.samples['sample_id']))
        tweet_ids = {v: k for k,v in tweet_ids.items()}
        self.samples['sample_id']=self.samples['sample_id'].apply(lambda x: tweet_ids[x])      
        self.data = list()
        for index, row in self.samples.iterrows():
            edge_1,edge_2 = row['edge_index']
            edge_1 = edge_1.astype('int64')
            edge_2 = edge_2.astype('int64')
            edges = torch.Tensor([edge_1,edge_2]).to(torch.int)
            x = torch.Tensor(row['x'])
            y = row['y']
            self.data.append(Data(x=x,y=y,edge_index=edges))
    def split_train_test(self,split):
        train_test_split = split
        train_len = int(len(self.samples) * train_test_split)
        self.train_data = self.data[:train_len]
        self.test_data = self.data[train_len:]
        self.train_loader = DataLoader(self.train_data, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=128, shuffle=False)
        self.X_train = [item.x[1] for item in self.train_data]
        self.y_train = [item.y for item in self.train_data]
        self.X_test = [item.x[1] for item in self.test_data]
        self.y_test = [item.y for item in self.test_data]
        self.X_train_np = [item.x[1].numpy().flatten() for item in self.train_data]
        self.y_train_np = [np.array(item.y).flatten() for item in self.train_data]
        self.X_test_np = [item.x[1].numpy().flatten() for item in self.test_data]
        self.y_test_np = [np.array(item.y).flatten() for item in self.test_data]
        self.X_train_stack = np.array(self.X_train_np).reshape(len(self.X_train_np),1,300)
        self.X_test_stack = np.array(self.X_test_np).reshape(len(self.X_test_np),1,300)
        self.y_train_stack = tf.stack(self.y_train_np)
        self.y_test_stack = tf.stack(self.y_test_np)

    def create_dataset(self,option='15'):
        self.data_dir = self.data_dir + ("/twitter15" if option == '15' else "/twitter16")
        df = pd.read_csv(self.data_dir+"/source_tweets.txt",names=["ID","x"],delimiter="\t",header=0)
        labels = pd.read_csv(self.data_dir+"/label.txt",names=["label","ID"],delimiter=":",header=0)
        df['y'] = labels['label'].apply(lambda x: 1 if x =='true' else 0 )
        df2 = df
        df2['edge_index'] = df['ID'].apply(lambda x: self.create_tree_for_tweet(x))
        vocab_size=3000
        sequence_length= 300
        vectorize_layer = layers.TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=sequence_length)
        vectorize_layer.adapt(df2['x'].values)
        text_vector_ds = tf.data.Dataset.from_tensor_slices(df2['x'].values).batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
        sequences = list(text_vector_ds.as_numpy_iterator())
        scaler = StandardScaler()
        new_seq = scaler.fit_transform(sequences)
        df2['x'] = df2['edge_index'].apply(lambda x: self.create_node_feature(x,new_seq))
        np.save(self.project_dir+"/processedData.npy",df2)
        
    def custom_standardization(self, input_data):
      lowercase = tf.strings.lower(input_data)
      return tf.strings.regex_replace(lowercase,
                                      '[%s]' % re.escape(string.punctuation), '')

    def create_node_feature(self,edge_index,new_seq):
        users = set(edge_index.flatten())
        values = new_seq[self.iterator]
        return_set = []
        for i in users:
            return_set.append(np.array(values))
        self.iterator = self.iterator + 1
        return np.array(return_set)
    
    def create_tree_for_tweet(self, tweet_id):
        tweet_edges = []
        edge_from = []
        edge_to = []
        all_edges =[]
        with open(self.data_dir+"/tree/"+str(tweet_id)+".txt") as f:
            f.readline()
            for line in f:
                a,b = line[:-1].replace('\'','').replace('[','').replace(']','').split('->')
                a = a.split(',')[0]
                b = b.split(',')[0]
                if (a=='ROOT') or (b=='ROOT'):
                    continue
                all_edges.append(a)
                all_edges.append(b)
                edge_from.append(a)
                edge_to.append(b)
        all_edges = set(all_edges)
        mapping = dict(enumerate(all_edges))
        mapping = {v: k for k,v in mapping.items()}
        for index,i in enumerate(edge_to):
            edge_to[index] = mapping[i]
        for index,i in enumerate(edge_from):
            edge_from[index]=mapping[i]
        tweet_edges = np.array([edge_from,edge_to])
        return tweet_edges

