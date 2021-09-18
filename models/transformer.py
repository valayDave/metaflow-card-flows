import torch.nn as nn
import pytorch_lightning as pl 


class ProductPredictionLayer(nn.Module):
    def __init__(self,embedding_size,num_products):
        super().__init__()
        self.product_prediction_layer = nn.Linear(
            embedding_size,num_products,
        )
        self.sfmx = nn.Softmax(dim=1)
    
    def forward(self,transformer_sequence):
        return self.sfmx(self.product_prediction_layer(
            transformer_sequence
        ))



class ProductRecommendationNet(pl.LightningModule):
    def __init__(self,num_products,embedding_size=512,num_heads = 8,num_layers=8) -> None:
        super().__init__()
        self.product_embedding = nn.Embedding(num_products,embedding_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads),num_layers=num_layers
        )
        self.product_pred_layer = ProductPredictionLayer(embedding_size,num_products)
    

    def forward(self,product_token_sequence,mask = None):
        sequence_embedding  = self.product_embedding(product_token_sequence)
        return self.product_pred_layer(self.transformer(sequence_embedding,mask =mask ))

    
    def training_step(self,batch,batch_nb) :
        # todo : fill steps
        pass

    def test_step(self,batch,batch_nb):
        # todo : fill steps
        pass
        
    