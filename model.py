import torch.nn as nn
from transformers import BertModel, GPT2LMHeadModel

#Defining the discriminator model
class Discreamial(nn.Module):
    def __init__(self, ):
        super(Discreamial, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-cased')
        self.Sequential = nn.Linear(1024, 2)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.Sequential(last_hidden_state_cls)
        return logits


#Using pre-trained GPT-2-medium
pre_Gen_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

#Defining the generator model
class Generator(nn.Module):
    def __init__(self,pre_Gen_model):
        super(Generator, self).__init__()
        self.GPT = pre_Gen_model.transformer
        self.Sequential = nn.Linear(1024, 2)

    def forward(self, input_ids,attention_mask):
        outputs = self.GPT(input_ids,attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.Sequential(last_hidden_state_cls)
        return logits