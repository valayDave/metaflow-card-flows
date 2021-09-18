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
        self.num_products = num_products
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads),num_layers=num_layers
        )
        self.product_pred_layer = ProductPredictionLayer(embedding_size,num_products)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    

    def forward(self,product_token_sequence,mask = None):
        sequence_embedding  = self.product_embedding(product_token_sequence)
        return self.product_pred_layer(self.transformer(sequence_embedding,mask =mask ))

    def _train_step(self,products,mask,labels):
        predictions = self(products,sequence_mask = mask)
        return predictions,self.loss_fn(predictions.view(-1, self.num_products),labels.view(-1))

    def training_step(self,batch,batch_nb) :
        # labels are MLM labels.
        products,mask,labels = batch
        predictions,loss = self._train_step(products,mask,labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self,batch,batch_nb) :
        # labels are MLM labels.
        products,mask,labels = batch
        predictions,loss = self._train_step(products,mask,labels)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self,batch,batch_nb):
        products,mask,labels = batch
        predictions,loss = self._train_step(products,mask,labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    