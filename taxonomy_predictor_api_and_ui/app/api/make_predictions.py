import torch
import torch.nn as nn
import os
import joblib
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, AdamW, BertConfig, BertTokenizer
from sentence_transformers import SentenceTransformer
dir_path = os.path.dirname(os.path.realpath(__file__))

sent_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
model_path_recommend = os.path.join(dir_path, '../models/model_euclidean_SENT_BERT_cos_attention_2_V3')
pathData = os.path.join(dir_path, '../data')

test_labels = joblib.load(pathData+'/test_labels')
test_labels = np.array(test_labels)

train_labels = joblib.load(pathData+'/train_labels')
train_labels = np.array(train_labels)

LE = joblib.load(pathData+'/label_encoder_reduced')

cos_label = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


labels_with_board = joblib.load(pathData+'/labels_formatted')

class MHSA(nn.Module):
  def __init__(self,
         emb_dim,
         kqv_dim,
         num_heads=2):
    super(MHSA, self).__init__()
    self.emb_dim = emb_dim
    self.kqv_dim = kqv_dim
    self.num_heads = num_heads

    self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)

  def forward(self, query, key, value):
    # print("query",query.shape)
    b, t = query.shape
    e = self.kqv_dim
    h = self.num_heads
    keys = self.w_k(key).view(b, h, e)
    values = self.w_v(value).view(b, h, e)
    queries = self.w_q(query).view(b, h, e)

    # keys = keys.transpose(2, 1)
    # queries = queries.transpose(2, 1)
    # values = values.transpose(2, 1)

    dot = queries @ keys.transpose(2, 1)  #(b*h*e) @ (b*e*h)
    dot = dot / np.sqrt(e)  # (b*h*h)
    dot = F.softmax(dot, dim=2)

    out = dot @ values   # (b*h*h) @ (b*h*e) = (b*h*e)
    out = out.contiguous().view(b, h * e)
    out = self.w_out(out)
    return out
    
class MulticlassClassifier(nn.Module):
    def __init__(self,bert_model_path):
        super(MulticlassClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path,output_hidden_states=True,output_attentions=False)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(576, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.act = torch.nn.ReLU()
        self.fc4 = nn.Linear(512, 1024)

        self.multi_head_attention = MHSA(1024, 64,16)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim = 1024,  num_heads = 4, batch_first=True)


    def forward(self,tokens,masks, targets=None, skip_attention=False):
        outputs = self.bert(tokens, attention_mask=masks)[2]
        output_1 = outputs[-1].permute(1,0,2)
        output_1 = torch.mean(output_1, dim=0)
        pooled_output = outputs[-1] # output_1 # torch.cat((output_1, output_2), dim=1)
        x = self.fc1(pooled_output)
        # x = self.fc2(x)
        
        targets_curr_batch = []
        for index_1, input_x in enumerate(x):
            # print(input_x.shape, torch.mean(input_x,dim=0).shape)
            distance = cos_label(torch.mean(input_x,dim=0).reshape(1,-1), unique_poincare_tensor)
            distances,indices = torch.topk(distance,1,largest=True)

            target_distances = (F.normalize(unique_poincare_tensor[indices],p=2,dim=1) - F.normalize(unique_poincare_tensor,p=2,dim=1)).pow(2).sum(1) #cos_label(unique_poincare_tensor[indices].reshape(1,-1), unique_poincare_tensor)
            distances,indices = torch.topk(target_distances,5,largest=False)
            targets_curr_batch.append(unique_poincare_tensor[indices].reshape(1,5,1024))

        targets_batch = torch.cat(targets_curr_batch, dim=0)

        attn_output, attn_output_weights = self.multihead_attn(targets_batch, x, x)

        x = torch.sum(attn_output,dim=1)
        return x



model = MulticlassClassifier('bert-base-uncased')
model.load_state_dict(torch.load(model_path_recommend+'/model_weights.zip',map_location=torch.device('cpu')))
recommender_tokenizer = BertTokenizer.from_pretrained(model_path_recommend, do_lower_case=True)

def get_labels(prediction):
    predicted_label =  LE.inverse_transform([prediction])
    return predicted_label[0]

def get_cleaned_taxonomy(taxonomy):
  cleaned_taxonomy = []
  for value in taxonomy:
      value = ' '.join(value.split(">>"))
      cleaned_taxonomy.append( value )
  return cleaned_taxonomy


def get_taxonomy_embeddings(labels):
    cleaned_taxonomy = get_cleaned_taxonomy(labels)
    taxonomy_vectors = sent_model.encode(cleaned_taxonomy)
    taxonomy_vectors = np.vstack(taxonomy_vectors)
    test_poincare_tensor = torch.tensor(taxonomy_vectors,dtype=torch.float)
    return test_poincare_tensor
test_poincare_tensor = get_taxonomy_embeddings(test_labels)
unique_poincare_tensor = get_taxonomy_embeddings(train_labels)
print("test_labels",test_poincare_tensor.shape)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def recommend_taxonomy(text):

    encoded_dict = recommender_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids = encoded_dict['input_ids']
    
    attention_masks = encoded_dict['attention_mask']

    # Tracking variables 
    predictions , true_labels = [], []
    with torch.no_grad():
        outputs = model(input_ids.reshape(1,-1),attention_masks.reshape(1,-1))
        
    # distances = (F.normalize(outputs,p=2,dim=1) - F.normalize(test_poincare_tensor,p=2,dim=1)).pow(2).sum(1)
    distances = cos(outputs,test_poincare_tensor)
    distances,indices = torch.topk(distances,3,largest=True)

    top_k_labels = test_labels[indices.cpu().numpy()]
    top_k_labels = list(top_k_labels)
    print("top_k_labels are", test_labels[indices.cpu().numpy()])


    final_list = []
    results = []
    for label,distance in zip(top_k_labels,distances):
        for formatted_label in labels_with_board:
            if label in formatted_label and len(label.split(">>"))>1:
                final_list.append((formatted_label,distance))
    results.append(text)
    
    for (prediction,distance) in final_list:

        if distance >=0.3:
            results.append({
                "taxonomy": prediction  ,
                "confidence": (0.5*distance+0.5 )     })
        else:
            results.append({"taxonomy":"None","confidence":0})
        


    return results
