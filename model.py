import transformers
import torch
from tqdm import tqdm
tqdm.pandas()
from sentence_transformers import SentenceTransformer
import pandas as pd
import tensorflow as tf

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions, attention_mask:tf.Tensor):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MetadataDistillationModel():
    """
    The model is initialized with a pretrained BERT model, which is then used to train the metadata distillation model.

    Parameters
    ----------
    name : str
        The name of the pretrained BERT model to be used. The model must be available in the HuggingFace model hub.
    Attributes
    ----------
    model : BertModel
        The pretrained BERT model.
    tokenizer : BertTokenizer
        The tokenizer used to tokenize the input sentences.
    sentence_transformer : SentenceTransformer
        The sentence transformer used to encode the input sentences.
    device : torch.device
        The device on which the model is trained.
    optimizer : torch.optim
        The optimizer used to train the model.
    loss_fn : torch.nn
        The loss function used to train the model.

    Methods
    -------
    save(path: str)
        Saves the model to the specified path.
    load(path: str)
        Loads the model from the specified path.
    eval()
        Sees the model architecture.
    encode_original(sentence: str)
        Encodes the input sentence using the sentence transformer.
    encode(sentence: str)
        Encodes the input sentence using the BERT model.
    to(device: torch.device)
        Sets the device on which the model is trained.
    """

    def __init__(self, name: str):
        self.model = transformers.BertModel.from_pretrained(name)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(name)
        self.sentence_transformer = SentenceTransformer(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save(self, path: str):
        self.model.save_pretrained(path+"-transformers")
        self.tokenizer.save_pretrained(path+"-transformers")
        self.sentence_transformer = SentenceTransformer(path+"-transformers")
        self.sentence_transformer.save(path)

    def load(self, path: str):
        self.model = transformers.BertModel.from_pretrained(path)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(path)
        self.sentence_transformer = SentenceTransformer(path)

    def eval(self):
        self.model.eval()

    def encode_original(self, sentence: str):
        return self.sentence_transformer.encode(sentence)

    def encode(self, sentence: str):
        encoded_input = self.tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        
        #transform tensor in array
        return sentence_embedding.squeeze(0).numpy()
    
    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)


    def forward(self, sentence: str):
        encoded_input = self.tokenizer(sentence,
                                       return_tensors='pt', 
                                       max_length=512,
                                       truncation=True,
                                       padding='max_length')
        #send to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        outputs = self.model(**encoded_input)
        
        y_hat_mean = mean_pooling(outputs, encoded_input['attention_mask'])
        y_hat = y_hat_mean.squeeze(0)
        #to return a tensor with 1024 dimensions only

        #return y_hat in device
        return y_hat.to(self.device)
    

    def train(self, x:pd.core.series.Series, y:torch.Tensor, epochs:int, lr:int = 1e-6, batch_size:int = 4):
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        loss_arr = []
        
        for i in range(epochs):
            #implement batches
            for j in tqdm(range(0, len(x), batch_size), total=len(x)//batch_size):
                self.optimizer.zero_grad()
                if j+batch_size >= len(x):
                    batch_x = x[j:]
                    batch_y = y[j:]

                    y_hat = self.forward(batch_x.tolist())
                    #send to device
                    batch_y = batch_y.to(self.device)
                    y_hat = y_hat.to(self.device)
                    loss = self.loss_fn(y_hat, batch_y)
                else:
                    batch_x = x[j:j+batch_size]
                    batch_y = y[j:j+batch_size]
                    
                    y_hat = self.forward(batch_x.tolist())
                    #send to device
                    batch_y = batch_y.to(self.device)
                    y_hat = y_hat.to(self.device)
                    loss = self.loss_fn(y_hat, batch_y)
                loss.backward()
                self.optimizer.step()
                loss_arr.append(loss.item())

            print(f"Epoch {i+1} loss: {loss.item()}")
        
        return loss_arr