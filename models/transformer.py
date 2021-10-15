import torch.nn as nn
import torch
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
    def __init__(self,num_products,embedding_size=256,num_heads =4,num_layers=4,learning_rate=1e-3) -> None:
        super().__init__()
        self.product_embedding = nn.Embedding(num_products,embedding_size)
        self.learning_rate = learning_rate
        self.num_products = num_products
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads,batch_first=True),num_layers=num_layers
        )
        self.product_pred_layer = ProductPredictionLayer(embedding_size,num_products)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def configure_optimizers(self):
        from transformers import AdamW

        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                        eps=1e-12, betas=(0.9, 0.999))
        return [optimizer]

    def forward(self,product_token_sequence,mask = None):
        sequence_embedding  = self.product_embedding(product_token_sequence)
        return self.product_pred_layer(self.transformer(sequence_embedding,src_key_padding_mask =mask ))

    def _train_step(self,products,mask,labels):
        predictions = self(products,mask = mask)
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
        

def train_transformer(dataset,\
                    logger_id,\
                    product_ids,
                    last_checkpoint_name = 'last_saved_model.pt',
                    transformer_args = dict(
                        embedding_size=256,
                        num_heads =4,
                        num_layers=4,
                        learning_rate=1e-3
                    ),
                    max_epochs=10,\
                    batch_size=64,):
    from . import dataloader
    from .dataloader import ProductTokenizer
    from .transformer import ProductRecommendationNet
    from pytorch_lightning import Trainer
    tokenizer = ProductTokenizer(product_ids)
    train_loader = dataloader.get_dataloader(
        dataset['train'],batch_size=batch_size,tokenizer=tokenizer
    )
    validation_loader = dataloader.get_dataloader(
        dataset['valid'],batch_size=batch_size,tokenizer=tokenizer
    )
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger("logs", name=logger_id)
    # trainer = Trainer(logger=logger)
    from pytorch_lightning.callbacks import ModelCheckpoint
    model_checkpoint = ModelCheckpoint(filename='model/checkpoints/{epoch:02d}-{val_loss:.2f}',
                            save_weights_only=True,
                            save_top_k=3,
                            monitor='validation_loss')
    
    gpu_dict = dict(gpus=torch.cuda.device_count())
    trainer_args = dict(
        max_epochs=max_epochs,\
        progress_bar_refresh_rate=25,\
        logger=logger,
        callbacks=[model_checkpoint]
    )
    if torch.cuda.device_count() > 0:
        if torch.cuda.device_count() > 1:
            gpu_dict.update(dict(
                accelerator='dp'
            ))
        trainer_args.update(gpu_dict)
    trainer = Trainer(
       **trainer_args
    )
    
    model = ProductRecommendationNet(
        len(product_ids),
        **transformer_args
    )
    trainer.fit(model,train_loader,validation_loader)
    print(f"Best Model Path : {model_checkpoint.best_model_path}")
    trainer.save_checkpoint(last_checkpoint_name)
    return model,logger.experiment.metrics,model_checkpoint.best_model_path