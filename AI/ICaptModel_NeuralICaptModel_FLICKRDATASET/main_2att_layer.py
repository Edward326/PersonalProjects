# Install dependencies using pip
import os










#NETWORK LAYOUT
#encoder
#cnn+relu+dropout to generate embedding for the image

#decoder
#1st attention layer->
#an multihead attention that uses as q,k,v embedding image from the encoder
#2nd attention layer->
#pass the output from the multihead attention to init_hidd_state of an bahdanau att,and as inputs of 
#encoder use the embedding of captions,
#then use for the init_hidd_state of decoder the context var of the encoder(last hidd_state),
#and use as inputs the embedding captions again
#outputs from the decoder will go through a linear layer(vocab_size)

#at test time the image will go to the encoder,the embedd of it will got rhiugh a multihead att,and 
#its output will go to the init_state of the encoder,the encoder takes as input a <bos>,then feeding the hiddenstate
#to an timestamp of decoder where as well there the input will be <bos>
#after that use the state generated by the decoder as input state of the next timestamp of the encoder and as input the 
#output from the decoder's last timestamp
#repeat until the <eos> is generated by the model itself or until noTokensToPredictMax tokens are reached
from colorama import Fore, Back, Style
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import spacy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests
import zipfile
import tarfile
import math
import time
#torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
embed_size,hidden_size = 512,1024
num_layers,learning_rate,num_epochs = 3,1e-4,10
default_dropout,num_heads=0.2,8
noTokensToPredictMax=40
batch_size=32
network_typeName="IC_NIC_CNN_MH_B"
spacy_en = spacy.load('en_core_web_sm')
raw_estimTime=5
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def plot_loss(train_loss, valid_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'bo', label='Train loss')
    plt.plot(epochs, valid_loss, 'ro', label='Valid loss')
    plt.title('Training and valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#donwload the dataset methods
def download(url, cache_dir=os.path.join('..', 'data')):
    """Download a file, return the local filename."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
        return fname
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(url, folder=None):
    """Download and extract a zip file."""
    fname = download(url, cache_dir=".")
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    fp = zipfile.ZipFile(fname, 'r')
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


#method to split the data downloaded into train.txt,validation.txt,test.txt
def split_data(all_captions_path, img_list_path, name):
    img_names = []
    with open(img_list_path, "r") as file:
        for line in file.readlines():
            img_name = line.strip("\n").split(".")[0]
            img_names.append(img_name)

    lines = []
    with open(all_captions_path, "r") as file:
        for line in file.readlines():
            words = line.replace(";",",").strip("\n").split()
            img_name = words[0].split(".")[0]

            if img_name in img_names:
                new_line = img_name + ".jpg;" + " ".join(words[1:])
                lines.append(new_line)

    with open(name, "w") as file:
        file.writelines("image;caption\n")
        lines = map(lambda x:x + '\n', lines)
        file.writelines(lines)


#method to create the vocabulary
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text]


#method to return the corpus
def load_captions(path):
    captions_list = []
    image_captions = {}
    with open(path, "r") as file:
        for line in file.readlines():
            words = line.strip("\n").split()
            caption = ' '.join(words[1:])
            captions_list.append(caption)

    return captions_list


#methods to tokenize the 3 files returned by split with the vocab created,shuffle them and orgainze in batches
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, sep=";")
        self.vocab = vocab
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<BOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class CollateDataset:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(root_folder,annotation_file,vocab,transform,num_workers=2,shuffle=True,pin_memory=True):
    dataset = FlickrDataset(root_folder, annotation_file, vocab, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CollateDataset(pad_idx=dataset.vocab.stoi["<PAD>"]),
    )
    return loader, dataset


#encoder of the model(CNN(resnet18 with all pooling layers//freezed)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(default_dropout)

    def forward(self, images):
        features = self.inception(images)
        if(self.training):
            features=features[0]#when is on training to have [batchsize,hiddensize]
        return self.dropout(self.relu(features))#produce the embedded for image


#decoder of the model,1st attention layer(multihead)
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries` is: (`batch_size`, no. of queries, `d`)
    # Shape of `keys` is: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values` is: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        
        batch_size, num_heads, _, head_dim = queries.shape
        queries = queries.reshape(batch_size * num_heads, 1, head_dim)#size 128,1,64
        keys = keys.reshape(batch_size * num_heads, 1, head_dim)
        values = values.reshape(batch_size * num_heads, 1, head_dim)
        # Use `keys.transpose(1,2)` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)#size 128,1,64

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X` is:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X` is:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
   

    # Reshape to (batch_size, seq_length, num_heads, head_dim)
    X = X.reshape(X.shape[0],X.shape[1], num_heads, -1)

    # Shape of output `X` is:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output` is:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X#batch_size,num_heads,1,64

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    #128,1,64
    X = X.reshape(X.shape[0]//num_heads, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)#32,1,4,64
    return X.reshape(X.shape[0],X.shape[1],hidden_size)

class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of `queries`, `keys`, or `values` is:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens` is:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values` is:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output` is: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat` is:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


#decoder of the model,2nd attention layer(bahdanau attention)
#the encoder of bahdanau
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,**kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
    
    def forward(self,init_state,X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        #init_state(32,256) X(32,27,256)
        # In RNN models, the first axis corresponds to time steps
        #x already permuted
        #init_state = init_state.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X,init_state)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        #torch.Size([24, 32, 256]) torch.Size([1, 32, 256])
        return output, state

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries` is: (`batch_size`, no.
        # of queries, 1, `num_hiddens`) and shape of `keys` is: (`batch_size`,
        # 1, no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores` is:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values` is: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

#decoder of bahdanau
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs` is: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]` is: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs` is: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]` is: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X` is: (`num_steps`, `batch_size`, `embed_size`)
        #X = X.permute(1, 0, 2)#X.shape=(32,notokens,256)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query` is: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context` is: (`batch_size`, 1, `num_hiddens`)
            #query(32,1,256) enc_outputs(32,notokens,256),enc_outputs(32,notokens,256)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)#32,1,256
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs` is:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))

        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

#class with the encoder and decoder of bahdanau combined
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,init_enc_state,enc_X, dec_X,init_decState=None,*args):
        if(init_decState!=None):
            init_enc_state=init_decState
        dec_state = self.decoder.init_state(self.encoder(init_enc_state,enc_X),None)

        return self.decoder(dec_X, dec_state)


#decoder of the model that intiialize an multihead(1st att layer) and an bahdnau(2nd att layer)
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderWithAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention1 =MultiHeadAttention(embed_size,embed_size,embed_size,hidden_size,
                                            num_heads,default_dropout)                                   
        self.attention2 = EncoderDecoder(Seq2SeqEncoder(vocab_size, embed_size, hidden_size,num_layers),
                                         Seq2SeqAttentionDecoder(vocab_size, embed_size, hidden_size,num_layers))
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(default_dropout)


    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))#embedding of captions
        #print(embeddings.shape)
        hiddens = []

        context1 = self.attention1(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
        context1 = context1.permute(1, 0, 2)
        context1=context1.repeat(num_layers, 1, 1) 
        context2,_= self.attention2(context1, embeddings, embeddings)

        return context2


#combine encoder(CNN) and decoder(MH+B) of the model
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderAttention= DecoderWithAttention(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderAttention(features, captions)
        return outputs

    #method used at testing
    def caption_image(self, image, vocabulary, max_length=noTokensToPredictMax):
        result_caption = []
        with torch.no_grad():
            features = self.encoderCNN(image)  # Extract features from the image
            features=features.unsqueeze(0)
            context = self.decoderAttention.attention1(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
            context = context.permute(1, 0, 2)
            context=context.repeat(num_layers, 1, 1) 
            # Initialize the decoder state with the context
            state = None
            inputs = torch.tensor([vocabulary.stoi["<BOS>"]]).to(image.device).unsqueeze(0)
            embeddings = self.decoderAttention.dropout(self.decoderAttention.embed(inputs))  # Initial embedding

            for _ in range(max_length):
                output, state = self.decoderAttention.attention2(context, embeddings, embeddings, state)
                state=state[1]
                predicted = output.squeeze(0).squeeze(0).argmax(0).item()
                result_caption.append(predicted)
            
                #if the model put itself the eos
                if vocabulary.itos[predicted] == "<EOS>":
                    break
            
                # Prepare the next input
                inputs = torch.tensor([predicted]).to(image.device).unsqueeze(0)
                embeddings = self.decoderAttention.dropout(self.decoderAttention.embed(inputs))  # Initial embedding

        return [vocabulary.itos[idx] for idx in result_caption]


#train method
def train(model, train_loader, val_loader,test_loader,optimizer, loss_criterion, device, num_epochs,alreadyParameters='ndef'):
    train_loss = []
    dev_loss = []

    #testing mode
    if(alreadyParameters=='already_learned'):
        test_run=[]
        print("Network type: "+Fore.MAGENTA+network_typeName+Style.RESET_ALL+"\n\n\n\n\n")
        print("TestDevice: set to    {}".format(device.type))
        if(device=='cuda'):
            print('GPU: ',Fore.MAGENTA,torch.cuda.get_device_name(device),Style.RESET_ALL)
        print("TESTING mode:\n\n")
        model.load_state_dict(torch.load('IC_NIC_CNN_MH_B__{}epochs_parameters.pth'.format(num_epochs)))
        model.eval()  # Set the model to train mode
        index = 0
        startExecTime=time.time()
        for idx, (imgs, captions) in enumerate(test_loader):
            with torch.no_grad():
                print("Phase: "+Fore.LIGHTGREEN_EX,f"{round(((index+1)/len(test_loader))*100,3)}%",Style.RESET_ALL,end='\r')
                imgs = imgs.to(device)
                captions = captions.to(device)
                outputs = model(imgs, captions)
                loss = loss_criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                test_run.append(loss.item())
                index += 1
        endExecTime=time.time()        
        # Compute average test loss
        avg_loss = np.mean(test_run)
        print("\n\nTest({} datasets): avgLoss is {}".format(len(test_loader)*batch_size,avg_loss))
        print(Fore.RED+Back.GREEN+"Total execution time: {:.2f}minutes".format((endExecTime-startExecTime)/60),Style.RESET_ALL)
        return

    
    #training mode
    print("Network type: "+Fore.MAGENTA+network_typeName+Style.RESET_ALL+"\n\n\n\n\n")
    print("TrainDevice: set to    {}".format(device.type))
    print('GPU: ',Fore.MAGENTA,torch.cuda.get_device_name(device),Style.RESET_ALL)
    print("TRAIN mode:\n\n")
    avgTrainEpochTime=0
    startExecTime=time.time()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        train_run_loss = []
        dev_run_loss = []
        index = 0
        #timeEpochReff=timeEpoch=0
        if epoch>0:
            timeEpochReff=avgTrainEpochTime*(num_epochs-epoch)
        else:
            timeEpochReff=(num_epochs-epoch)*raw_estimTime
        timeEpoch=timeEpochReff/60
        
        startExecTime2=time.time()
        for idx, (imgs, captions) in enumerate(train_loader):
            startExecTime3=time.time()
            hehe=int((timeEpoch-int(timeEpoch))*60)
            print(f"Epoch_{epoch+1} ,Phase: "+Fore.LIGHTGREEN_EX,f"{round((((index+1)/len(train_loader))*((epoch+1)/num_epochs))*100,3)}% ",Fore.RED,'   ETA: {}:{} hours'.format(int(timeEpoch),hehe),Style.RESET_ALL,end='\r')
            imgs = imgs.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, captions)
            loss = loss_criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            loss.backward()
            optimizer.step()
            train_run_loss.append(loss.item())
            index += 1
            endExecTime3=time.time()
            diff=float((endExecTime3-startExecTime3)/60/60)
            timeEpoch-=diff
               
        # Compute average training loss for the epoch
        avg_loss = np.mean(train_run_loss)
        train_loss.append(avg_loss)


        # Validation loop
        index = 0
        model.eval()  # Set the model to evaluation mode
        print(f"Epoch_{epoch+1} ,test phase on validation\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",end='\r')
        for idx, (imgs, captions) in enumerate(val_loader):
            with torch.no_grad():
                imgs = imgs.to(device)
                captions = captions.to(device)
                outputs = model(imgs, captions)
                loss = loss_criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                dev_run_loss.append(loss.item())
                index += 1

        # Compute average validation loss for the epoch
        endExecTime2=time.time()
        avgTrainEpochTime=float((endExecTime2-startExecTime2)/60)
        avg_loss = np.mean(dev_run_loss)
        dev_loss.append(avg_loss)
        print(f'Epoch_{epoch+1}--> ',Fore.GREEN,'DONE\t\t\t\t\t\t\t\t\t\t\t\t',Style.RESET_ALL)
        #print("\n")
    endExecTime=time.time()
    torch.save(model.state_dict(),'IC_NIC_CNN_MH_B__{}epochs_parameters.pth'.format(num_epochs))#save the parameters     
    print("\n\nTrain({} datasets): avgLoss is {}".format(len(train_loader)*batch_size,np.mean(train_loss)))
    print("Validation({} datasets): avgLoss is {}".format(len(val_loader)*batch_size,np.mean(dev_loss)))
    print(Fore.RED+Back.GREEN+"Total execution time:{}minutes".format(float((endExecTime-startExecTime)/60)),Style.RESET_ALL)
    plot_loss(train_loss,dev_loss)
    return


#method to predict the first image
def print_example(model, device, loader, dataset):
    model.eval()
    img, caption = next(iter(loader))

    # Move data to the same device as the model
    img = img.to(device)
    caption = caption.to(device)

    caption = caption.transpose(0, 1)
    imgreff = img[0].permute(1, 2, 0).cpu().numpy()  # Move image to CPU for visualization

    if imgreff.min() < 0 or imgreff.max() > 1:
        imgreff = (imgreff - imgreff.min()) / (imgreff.max() - imgreff.min())  # Normalize to [0, 1]
    
    plt.imshow(imgreff)
    plt.show()

    print("Target CAPTION: " + " ".join([dataset.vocab.itos[idx] for idx in caption[0].tolist() if dataset.vocab.itos[idx] not in {"<PAD>", "<BOS>", "<EOS>"}]))
    print("\nPredicted CAPTION: " + " ".join(model.caption_image(img[0].unsqueeze(0), dataset.vocab)))





if __name__ == '__main__':
    os.system("pip install torch")
    os.system("pip install torchvision")
    os.system("pip install Pillow")
    os.system("pip install spacy")
    os.system("pip install pandas")
    os.system("pip install matplotlib")
    os.system("pip install requests")
    os.system("pip install colorama")
    os.system("python3 -m spacy download en_core_web_sm")
    os.system('cls')
    #=========================================
    #create de dataset
    #download the flockr8k dataset:the images with their captions(corpus to be used by the vocab when its created)
    download_extract('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', 'Flickr8k_Dataset')
    download_extract('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', 'Flickr8k_text')
    split_data("Flickr8k.token.txt", "Flickr_8k.trainImages.txt", "train.txt")
    split_data("Flickr8k.token.txt", "Flickr_8k.devImages.txt", "validation.txt")
    split_data("Flickr8k.token.txt", "Flickr_8k.testImages.txt", "test.txt")
    #create a vocab on the corpus(captions of the images)
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(load_captions("Flickr8k.token.txt"));
    #use the vocab to tokenize the corpus(captions of the images) and separate them to iter datasets:train,val,test
    transform = transforms.Compose(
        [transforms.Resize((299, 299)),
         transforms.ToTensor(),
         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
         ])
    train_loader, train_dataset = get_loader(
    "Flicker8k_Dataset", "train.txt", vocab, transform=transform)
    val_loader, val_dataset = get_loader(
    "Flicker8k_Dataset", "validation.txt", vocab, transform=transform)
    test_loader, test_dataset = get_loader(
    "Flicker8k_Dataset", "test.txt", vocab, transform=transform, shuffle=False)

    #define the network
    vocab_size = len(vocab)#2993
    device = try_gpu()
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    #deactivate the learning process of the cnn from encoder(pretrained already)-->FREEZE
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    #train the NIC-NeuralImageCaptioning aka 'Show and tell'  ,model for image captioning problem
    train(model,train_loader,val_loader,test_loader,optimizer,loss_criterion,device,num_epochs)

    #test the trained network
    #print_example(model, device, test_loader, test_dataset)