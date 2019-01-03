
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
nltk.download('punkt')
from PIL import Image
from pycocotools.coco import COCO
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence



class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


vocab = build_vocab(json='data/annotations/captions_train2014.json', threshold=int(4))
vocab_path = 'data/vocab.pkl'
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print("Total vocabulary size: {}".format(len(vocab)))
print("Saved the vocabulary wrapper to '{}'".format(vocab_path))



class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20,net='gru'):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if net == 'lstm':
            self.unit = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif net == 'gru':
            self.unit = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        elif net == 'elman':
            self.unit = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            print('Choose one of the three decoder models: lstm, gru or elman')
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.unit(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.unit(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

#     def beam_search_decoder(self,features, k):
#         sequences = [[list(), 1.0]]
#         # walk over each step in sequence
#         for row in features:
#             all_candidates = list()
#             # expand each current candidate
#             for i in range(len(sequences)):
#                 seq, score = sequences[i]
#                 for j in range(len(row)):
#                     try:
#                         candidate = [seq + [j], score * -log(row[j])]
#                         all_candidates.append(candidate)
#                     except ValueError:
#                         candidate = [seq + [j], 0]
#                         all_candidates.append(candidate)
#             # order all candidates by score
#             ordered = sorted(all_candidates, key=lambda tup:tup[1])
#             # select k best
#             sequences = ordered[:k]
#         return sequences




def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))



image_dir = 'data/train2014/'
output_dir = 'data/resized2014/'
image_size = [256,256]
resize_images(image_dir, output_dir, image_size)


image_dir = 'data/val2014/'
output_dir = 'data/resizedval2014/'
image_size = [256,256]
resize_images(image_dir, output_dir, image_size)


image_dir = 'data/test2014/'
output_dir = 'data/resizedtest2014/'
image_size = [256,256]
resize_images(image_dir, output_dir, image_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print('Yes')


transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)


data_loader = get_loader('data/resized2014/', 'data/annotations/captions_train2014.json', vocab, 
                             transform, 128,
                             shuffle=True, num_workers=2) 


encoder = EncoderCNN(512).to(device)
decoder = DecoderRNN(512, 1024, len(vocab), 1,net = 'gru').to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)



# Train the models
epochs = 3
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
            
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
        # Save the model checkpoints
        if (i+1) % 1500 == 0:
            torch.save(decoder.state_dict(), os.path.join(
                    'models/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                    'models/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


# ## Evaluation


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)


encoder = EncoderCNN(512).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = DecoderRNN(512, 1024, len(vocab), 1,net='gru')
# encoder = encoder.to(device)
# decoder = decoder.to(device)


encoder.load_state_dict(torch.load('models/encoder-3-3000.ckpt'))
decoder.load_state_dict(torch.load('models/decoder-3-3000.ckpt'))
encoder = encoder.to(device)
decoder = decoder.to(device)


import os
import random

def getRandomFile(path):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  files = os.listdir(path)
  index = random.randrange(0, len(files))
  return files[index]


# ## Testing Metrics


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

import time
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderCNN(512).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = DecoderRNN(512, 1024, len(vocab), 1,net ='gru')
encoder.load_state_dict(torch.load('models/encoder-2-3000.ckpt'))
decoder.load_state_dict(torch.load('models/decoder-2-3000.ckpt'))
encoder = encoder.to(device)
decoder = decoder.to(device)

total = 0
scores_bleu = 0
scores_bleu1 = 0
scores_bleu2 = 0
scores_bleu3 = 0
scores_bleu4 = 0
scores_rouge = 0


from operator import itemgetter
root = 'data/resizedval2014/'
json = 'data/annotations/captions_val2014.json'
coco = COCO(json)
ids = list(coco.anns.keys()) 
list_im = []
for index in range(len(ids)):
    ann_id = ids[index]
    caption = coco.anns[ann_id]['caption']
    img_id = coco.anns[ann_id]['image_id']
    path = coco.loadImgs(img_id)[0]['file_name']
    a = [path,caption]
    list_im.append(a)
    
sorted_list=sorted(list_im,key=itemgetter(0))
flag=sorted_list[0][0]
img_cap=[]
app_cap=[]

for i in sorted_list:
    if(i[0]==flag):
        app_cap.append(i[1])
    else:
        img_cap.append([flag,app_cap])
        app_cap=[]
        flag=i[0]
        app_cap.append(i[1])
img_cap.append([flag,app_cap])


#Calculate Rouge score

import numpy as np
import pdb

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]



def Rouge_score(candidate, refs):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    assert(len(candidate)==1)
    assert(len(refs)>0)         
    prec = []
    rec = []

    # split into tokens
    token_c = candidate[0].split(" ")
    
    for reference in refs:
        # split into tokens
        token_r = reference.split(" ")
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs/float(len(token_c)))
        rec.append(lcs/float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)
    beta = 1.2
    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
    else:
        score = 0.0
    return score

##Meteor score
def meteor_score_from_files(ref, hyp, scores_file=None):
        """
            Source: https://www.cs.cmu.edu/~alavie/METEOR/examples.html
            Java -jar command: java -Xmx2G -jar meteor-*.jar [hyp.txt] [ref.txt] -norm -f system1 > test.txt
            Command to obtain more results:
                'java -Xmx2G -jar meteor-*.jar example/xray/system1.hyp example/xray/reference -norm -writeAlignments -f system1'
        :param ref: file containing reference text
        :param hyp: file containing hypotheses text
        :param scores_file: file to store METEOR score
        """

        if scores_file is None:
            scores_file = 'Flickr8k_text/test_meteor.txt'

        os.system(
            'java -Xmx2G -jar {dir}meteor-*.jar {hyp_file} {ref_file} -norm -f system1 > {scores_file}'.
            format(dir='Flickr8k_text/', hyp_file=hyp, ref_file=ref, scores_file=scores_file))


file = getRandomFile('data/resizedval2014/')
image = load_image('data/resizedval2014/'+ str(file), transform)
print(image.shape)
image_tensor = image.to(device)
print(image_tensor.shape)
feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()
sampled_caption = []
for word_id in sampled_ids:
    word = vocab.idx2word[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
sentence = ' '.join(sampled_caption)
print (sentence)
image = Image.open('data/resizedval2014/'+ str(file))
plt.imshow(np.asarray(image))


start = time.time()
for i in range(len(img_cap)):
    image_dir = img_cap[i][0]
    captions = img_cap[i][1]
    actual = []
    for j in captions:
        k = nltk.tokenize.word_tokenize(str(j).lower())
        actual.append(k)
        
    image = load_image('data/resizedval2014/'+ image_dir, transform)
    #print(image.shape)
    
    image_tensor = image.to(device)
    #print(image_tensor.shape)
    if(list(image.shape)==[1, 1, 224, 224]):
        continue
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []
    for i,word_id in enumerate(sampled_ids):
        if(i>0):
            word = vocab.idx2word[word_id]
            if(word == '<unk>'):
                word = 'abc'
            if(word == '<end>'):
                break
            sampled_caption.append(word)
    sentence = ' '.join(sampled_caption)
    generated = nltk.tokenize.word_tokenize(str(sentence).lower())
    
    #Bleu Score                                                                                                                                                                                         
    score = sentence_bleu(actual,generated)
    scores_bleu += score
    #print(score)
    sc1 = sentence_bleu(actual,generated,weights=(1,0,0,0))
    scores_bleu1 += sc1
    #print(sc1)
    sc2 = sentence_bleu(actual,generated,weights=(0,1,0,0))
    scores_bleu2 += sc2
    #print(sc2)
    sc3 = sentence_bleu(actual,generated,weights=(0,0,1,0))
    scores_bleu3 += sc3
    #print(sc3)
    sc4 = sentence_bleu(actual,generated,weights=(0,0,0,1))
    scores_bleu4 += sc4
    total += 1
    #print(sc4)
    total += 1
    sc5 = Rouge_score([sentence],[captions[0]])
    scores_rouge += sc5 
    #print(sc5)
    #print("\n")
    
print("Bleu Score: {}".format(scores_bleu/total))
print("Bleu1 Score: {}".format(scores_bleu1/total))
print("Bleu2 Score: {}".format(scores_bleu2/total))
print("Bleu3 Score: {}".format(scores_bleu3/total))
print("Bleu4 Score: {}".format(scores_bleu4/total)) 
print("Rouge Score: {}".format(scores_rouge/total)) 
#print("Meteor Score: {}".format(scores_meteor/total))
end = time.time()
print(end-start)

scores_meteor_file = "Score_MSCOCO_Meteor.txt"
ref_file='Flickr8k_text/Reference_caption_COCO.txt'
hyp_file='Flickr8k_text/Generated_caption_COCO.txt'
meteor = meteor_score_from_files(ref_file, hyp_file, scores_file=scores_meteor_file)
print(meteor)