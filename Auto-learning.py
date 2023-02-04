import torch
from transformers import GPT2Tokenizer
import pandas as pd
from data_pre_processing import clean_data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from model import Generator,pre_Gen_model
from initialize_model import initialize
from trainer import train

device = torch.device('cuda:0')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv(r'Gen_disc.csv')
train_data_text = clean_data(df['Text'])
train_data_label = df['label']

encoded_data_train = tokenizer.batch_encode_plus(
    train_data_text,
    max_length=45, 
    truncation=True,
    padding = True
)

input_ids_train = torch.tensor(encoded_data_train['input_ids'])
input_attention_mask_train = torch.tensor(encoded_data_train['attention_mask'])
labels_train = torch.tensor(train_data_label).long()

dataset_train = TensorDataset(input_ids_train, input_attention_mask_train, labels_train)
dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=64)

Generators_model, optimizer, scheduler = initialize(Generator,device,dataloader_train,lr=3e-5,eps=1e-8,epochs=1,supervision=True)
Generators_model.load_state_dict(torch.load('Generators_model.pt'))

train(Generators_model,device,dataloader_train,optimizer,scheduler, dataloader_train, epochs=1, evaluation=False)
torch.save(Generators_model.state_dict(),'Generators_model.pt')
torch.save(pre_Gen_model.state_dict(),'pre_Gen_model.pt')