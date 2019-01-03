
# coding: utf-8

# In[79]:


import os
import io
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from PIL import Image
import nltk
#nltk.download('punkt')
import itertools

import time
import matplotlib.pyplot as plt


# In[8]:


folder='/home/ubuntu/flickr30k_image/'


# In[9]:


img_list=os.listdir(folder)


# In[11]:


lst=[]
with open('/home/ubuntu/flickr30k_text/results_20130124.token','r') as f:
    for i in f:
        lst.append(i.strip())


# In[12]:


##Store all the captions for each image
annot={}
for i in range(0,len(lst),5):
    ann=[]
    t1=lst[i]
    for j in range(i,i+5):
        tmp=lst[j]
        tmp=tmp.split('\t')
        ann.append([tmp[1].lower()])
    t1=t1.split('\t')
    annot[t1[0].split('#')[0]]=ann


# In[13]:


cap=[]
for i in range(0,len(lst),5):
    j=lst[i].split('\t')
    cap.append([j[0].split('#')[0],j[1]])


# In[14]:


train_list=cap[:]


# In[15]:


train_img_list=[i[0] for i in train_list]


# In[16]:


train_dict={i[0]:i[1] for i in train_list}


# In[17]:


##Tokenize train captions
train_token=[]
train_tok=[]
for (j,i) in train_dict.items():
    train_token.append([j,nltk.word_tokenize(i.lower())])
    train_tok.append(nltk.word_tokenize(i.lower()))


# In[18]:


##LOAD IMAGES Flickr8k

##Training images list
train_img_list8=[]
with open('Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
    for i in f:
        train_img_list8.append(i.strip())

##Test images list
test_img_list=[]
with open('Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
    for i in f:
        test_img_list.append(i.strip())


# In[19]:


img_caption=[]
with open('Flickr8k_text/Flickr8k.token.txt','r') as f:
    for i in f:
        img_caption.append(i)


# In[20]:


##Store all the captions for each image
annot={}
for i in range(0,len(img_caption),5):
    ann=[]
    t1=img_caption[i].strip()
    for j in range(i,i+5):
        tmp=img_caption[j].strip()
        tmp=tmp.split('\t')
        ann.append([tmp[1].lower()])
    t1=t1.split('\t')
    annot[t1[0].split('#')[0]]=ann


# In[21]:


##Caption and Image List
cap_dict={}
for i in range(0,len(img_caption),5):
    tmp=img_caption[i].strip()
    tmp=tmp.split('\t')
    cap_dict[tmp[0].split('#')[0]]=tmp[1].lower()


# In[22]:


##Training captions
train_cap_dict8={}
for i in train_img_list8:
    train_cap_dict8[i]=cap_dict[i]


# In[23]:


##Test captions
test_cap_dict={}
for i in test_img_list:
    test_cap_dict[i]=cap_dict[i]


# In[24]:


##Tokenize train captions
train_token8=[]
train_tok8=[]
for (j,i) in train_cap_dict8.items():
    train_token8.append([j,nltk.word_tokenize(i)])
    train_tok8.append(nltk.word_tokenize(i))


# In[25]:


##Tokenize test captions
test_token=[]
for (j,i) in test_cap_dict.items():
    test_token.append([j,nltk.word_tokenize(i)])


# In[26]:


##word_to_id and id_to_word
all_tokens = itertools.chain.from_iterable(train_tok)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(train_tok)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)


# In[27]:


##Sort the indices by word frequency

train_token_ids = [[word_to_id[token] for token in x[1]] for x in train_token]
count = np.zeros(id_to_word.shape)
for x in train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]


# In[28]:


##Recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}


# In[29]:


print("Vocabulary size: "+str(len(word_to_id)))


# In[30]:


## assign -4 if token doesn't appear in our dictionary
## add +4 to all token ids, we went to reserve id=0 for an unknown token
train_token_ids = [[word_to_id.get(token,-4)+4 for token in x[1]] for x in train_token]

train_token_ids8 = [[word_to_id.get(token,-4)+4 for token in x[1]] for x in train_token8]
test_token_ids = [[word_to_id.get(token,-4)+4 for token in x[1]] for x in test_token]


# In[31]:


word_to_id['<unknown>']=-4
word_to_id['<start>']=-3
word_to_id['<end>']=-2
word_to_id['<pad>']=-1

for (_,i) in word_to_id.items():
    i+=4
    word_to_id[_]=i


# In[32]:


id_to_word_dict={}
cnt=4
for i in id_to_word:
    id_to_word_dict[cnt]=i
    cnt+=1
id_to_word_dict[0]='<unknown>'
id_to_word_dict[1]='<start>'
id_to_word_dict[2]='<end>'
id_to_word_dict[3]='<pad>'


# In[34]:


##Length of each caption
train_cap_length={}
for i in train_token:
    train_cap_length[i[0]]=len(i[1])+2
    
train_cap_length8={}
for i in train_token8:
    train_cap_length8[i[0]]=len(i[1])+2
    

test_cap_length={}
for i in test_token:
    test_cap_length[i[0]]=len(i[1])+2


# In[35]:


##Add <start> and <end> tokens to each caption
for i in train_token_ids:
    i.insert(0,word_to_id['<start>'])
    i.append(word_to_id['<end>'])

for i in train_token_ids8:
    i.insert(0,word_to_id['<start>'])
    i.append(word_to_id['<end>'])
    
for i in test_token_ids:
    i.insert(0,word_to_id['<start>'])
    i.append(word_to_id['<end>'])


# In[36]:


##Pad train captions
length=[]
for (i,j) in train_cap_length.items():
    length.append(j)
max_len=max(length)

for n,i in enumerate(train_token):
    if (train_cap_length[i[0]] < max_len):
        train_token_ids[n].extend(word_to_id['<pad>'] for i in range(train_cap_length[i[0]],max_len))
        


# In[37]:


##Convert token ids to dictionary for train
train_token_ids_dict={}
for n,i in enumerate(train_token):
    train_token_ids_dict[i[0]]=train_token_ids[n]


# In[38]:


##Pad train captions_8k
length=[]
for (i,j) in train_cap_length8.items():
    length.append(j)
max_len=max(length)

for n,i in enumerate(train_token8):
    if (train_cap_length8[i[0]] < max_len):
        train_token_ids8[n].extend(word_to_id['<pad>'] for i in range(train_cap_length8[i[0]],max_len))
        


# In[39]:


##Convert token ids to dictionary for train_8k
train_token_ids_dict8={}
for n,i in enumerate(train_token8):
    train_token_ids_dict8[i[0]]=train_token_ids8[n]


# In[40]:


##Pad test captions
length=[]
for (i,j) in test_cap_length.items():
    length.append(j)
max_len=max(length)

for n,i in enumerate(test_token):
    if (test_cap_length[i[0]] < max_len):
        test_token_ids[n].extend(word_to_id['<pad>'] for i in range(test_cap_length[i[0]],max_len))


# In[41]:


##Convert token ids to dictionary for test
test_token_ids_dict={}
for n,i in enumerate(test_token):
    test_token_ids_dict[i[0]]=test_token_ids[n]


# In[42]:


## save dictionary
np.save('flickr30k_text/flickr30k_dictionary.npy',np.asarray(id_to_word))


# In[43]:


class Dataset(data.Dataset):
    def __init__(self,img_dir,img_id,cap_dictionary,cap_length,transform=None):
        self.img_dir=img_dir
        self.img_id=img_id
        self.transform=transform
        self.cap_dictionary=cap_dictionary
        self.cap_length=cap_length
    
    def __len__(self):
        return len(self.img_id)
    
    def __getitem__(self,index):
        img=self.img_id[index]
        img_open=Image.open(self.img_dir+img).convert('RGB')
        
        if self.transform is not None:
            img_open=self.transform(img_open)
        
        cap=np.array(self.cap_dictionary[img])
        cap_len=self.cap_length[img]
        
        return img_open,cap,cap_len


# In[44]:


transform_train = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])


# In[45]:


img_dir='flickr30k_resized_image/'
img_dir1='Flickr8k_resized_image/'
train_data=Dataset(img_dir,train_img_list,train_token_ids_dict,train_cap_length,transform_train)

train_data8=Dataset(img_dir1,train_img_list8,train_token_ids_dict8,train_cap_length8,transform_train)
test_data=Dataset(img_dir1,test_img_list,test_token_ids_dict,test_cap_length,transform_test)


# In[46]:


train_dataloader=data.DataLoader(train_data,batch_size=32, shuffle=True, num_workers=2)

train_dataloader8=data.DataLoader(train_data8,batch_size=32, shuffle=True, num_workers=2)
test_dataloader=data.DataLoader(test_data,batch_size=32, shuffle=True, num_workers=2)


# In[47]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[48]:


if torch.cuda.is_available():
    print('Yes')


# In[49]:


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


# In[54]:


encoder = EncoderCNN(1024)
decoder = DecoderRNN(1024, 1024, len(word_to_id), 1)


# In[55]:


##Function to sort the captions and images according to caption length
def sorting(image,caption,length):
    srt=length.sort(descending=True)
    image=image[srt[1]]
    caption=caption[srt[1]]
    length=srt[0]
    
    return image,caption,length


# In[56]:


# Loss and optimizer
encoder.to(device)
decoder.to(device)

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)


# In[57]:


##Train the model
encoder.train()
decoder.train()

train_loss=[]
time1=time.time()
epochs=3
total_step=len(train_dataloader)
for epoch in range(epochs):
    for i, (images,captions,lengths) in enumerate(train_dataloader):
        
        if((len(captions)!=32)):
            continue
        
        images = images.to(device)
        captions = captions.to(device)
        
        images,captions,lengths=sorting(images,captions,lengths)
        
        targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]
        
        
        ##Forward,backward and optimization
        features = encoder(images)
        outputs = decoder(features,captions,lengths)
        loss = criterion(outputs,targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss)
        
        # Print log info
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
print('RUNNING TIME: {}'.format(time.time()-time1))  
print('PERPLEXITY: '+str(np.exp(loss.item())))


# In[161]:


features8=Variable(features,requires_grad=False)


# In[162]:


class DecoderRNN8(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN8, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


# In[163]:


decoder8 = DecoderRNN8(1024, 1024, len(word_to_id), 1)
decoder8.to(device)


# In[164]:


criterion8 = nn.CrossEntropyLoss()
optimizer8 = torch.optim.Adam(decoder8.parameters(), lr=0.001)


# In[165]:


##Flickr8k
##Train the model

decoder8.train()

train_loss=[]
time1=time.time()
epochs=5

total_step=len(train_dataloader8)
for epoch in range(epochs):
    for i, (images,captions,lengths) in enumerate(train_dataloader8):
        
        if((len(captions))!=32):
            continue
        
        images = images.to(device)
        captions = captions.to(device)
        
        images,captions,lengths=sorting(images,captions,lengths)
        
        targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]
        
        
        ##Forward,backward and optimization
        outputs = decoder8(features8,captions,lengths)
        loss8 = criterion8(outputs,targets)
        decoder8.zero_grad()
        loss8.backward()
        optimizer8.step()
        
        train_loss.append(loss8)
        
        # Print log info
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, epochs, i, total_step, loss8.item(), np.exp(loss8.item()))) 
                
print('RUNNING TIME: {}'.format(time.time()-time1))  
print('PERPLEXITY: '+str(np.exp(loss8.item())))


# In[101]:


torch.save(encoder,os.path.join('models/transfer/','encoder_transfer.model'))
torch.save(decoder,os.path.join('models/transfer/','decoder_transfer.model'))


# In[141]:


with open('flickr30k_text/train_loss_transfer.txt','w') as f:
    for i in train_loss:
        f.write(str(i.item()))
        f.write(' ')


# In[142]:


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


# In[143]:


encoder.eval()


# In[144]:


decoder.eval()


# In[145]:


encoder.to(device)
decoder.to(device)


# In[146]:


import random

def getRandomFile(img_list):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  #files = os.listdir(path)
  ind = random.randrange(0, len(img_list))
  return img_list[ind]


# In[147]:


file = getRandomFile(test_img_list)
image = load_image('Flickr8k_resized_image/'+ str(file), transform_test)
image_tensor = image.to(device)
feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()
sampled_caption = []
for word_id in sampled_ids:
    word = id_to_word_dict[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
sentence = ' '.join(sampled_caption)
print (sentence)
image = Image.open('Flickr8k_resized_image/'+ str(file))
plt.imshow(np.asarray(image))


# In[150]:


##Test the model
encoder.eval()
decoder8.eval()

test_loss=[]
time1=time.time()

total_step=len(test_dataloader)
for i, (images,captions,lengths) in enumerate(test_dataloader):
        
    images = images.to(device)
    captions = captions.to(device)
        
    images,captions,lengths=sorting(images,captions,lengths)
        
    targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]
        
    with torch.no_grad():    
        ##Forward,backward and optimization
        features = encoder(images)
        outputs = decoder8(features,captions,lengths)
        loss = criterion(outputs,targets)
        
    test_loss.append(loss)
        
    # Print log info
    if i % 100 == 0:
        print('Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                 .format(i, total_step, loss.item(), np.exp(loss.item()))) 
                
print('RUNNING TIME: {}'.format(time.time()-time1))
print('PERPLEXITY: {}'.format(np.exp(loss.item())))


# In[151]:


with open('flickr30k_text/test_loss_transfer.txt','w') as f:
    for i in test_loss:
        f.write(str(i.item()))
        f.write(' ')


# In[153]:


#Bleu score (new)
from nltk.translate.bleu_score import sentence_bleu

device = 'cuda'
encoder.to('cuda')
decoder8.to('cuda')
encoder.eval()
decoder8.eval()

total = 0
scores_bleu = 0 
scores_bleu1 = 0 
scores_bleu2 = 0 
scores_bleu3 = 0 
scores_bleu4 = 0 
with torch.no_grad():
    generated_caption=[]
    for i in range(len(test_img_list)):
        
        ##Generated caption
        file=test_img_list[i]
        image = load_image('Flickr8k_resized_image/'+ str(file), transform_test)
        image_tensor = image.to(device)
        feature = encoder(image_tensor)
        sampled_ids = decoder8.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = id_to_word_dict[word_id]
            if word == '<end>':
                break
            sampled_caption.append(word)
        sampled_caption.remove('<start>')
        sentence = ' '.join(sampled_caption)
        generated_caption.append([file,sentence])
        

        ##Actual caption
        org_sentence=annot[file]
        
        ##Tokenize the captions
        generated=nltk.word_tokenize(str(sentence))
        #actual=nltk.word_tokenize(str(org_sentence))
        
        actual=[]
        for i in org_sentence:
            #print(nltk.word_tokenize(i[0]))
            actual.append(nltk.word_tokenize(i[0]))   

        
        
        ##Bleu score
        score = sentence_bleu(actual,generated)
        scores_bleu += score
        sc1 = sentence_bleu(actual,generated,weights=(1,0,0,0))
        scores_bleu1 += sc1
        sc2 = sentence_bleu(actual,generated,weights=(0,1,0,0))
        scores_bleu2 += sc2        
        sc3 = sentence_bleu(actual,generated,weights=(0,0,1,0))
        scores_bleu3 += sc3
        sc4 = sentence_bleu(actual,generated,weights=(0,0,0,1))
        scores_bleu4 += sc4
        total +=1
        
print("Bleu Score: {}".format(scores_bleu/total))
print("Bleu1 Score: {}".format(scores_bleu1/total))
print("Bleu2 Score: {}".format(scores_bleu2/total))
print("Bleu3 Score: {}".format(scores_bleu3/total))
print("Bleu4 Score: {}".format(scores_bleu4/total))
        
        


# In[154]:


with open('flickr30k_text/ref_transfer.txt','w') as f:
    for i in generated_caption:
        f.write(annot[i[0]][0][0])
        f.write('\n')


# In[155]:


with open('flickr30k_text/hyp_transfer.txt','w') as f:
    for i in generated_caption:
        f.write(i[1])
        f.write('\n')


# In[156]:


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


# In[157]:


scores_meteor_file = "flickr30k_text/test_transfer_meteor.txt"


# In[158]:


ref_file='flickr30k_text/ref_transfer.txt'
hyp_file='flickr30k_text/hyp_transfer.txt'
meteor_score_from_files(ref_file, hyp_file, scores_file=scores_meteor_file)


# In[159]:


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


# In[160]:


score_list=[]
for i in generated_caption:
    candidate=[i[1]]
    refs=annot[i[0]][0]
    score_list.append(Rouge_score(candidate,refs))
print("Rouge score: "+str(sum(score_list)/len(generated_caption)))

