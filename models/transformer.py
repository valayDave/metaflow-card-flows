import torch.nn as nn
import torch
import pytorch_lightning as pl 
import math

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



# thank You : https://github.com/yaohungt/Multimodal-Transformer/blob/master/modules/position_embedding.py
# Code adapted from the fairseq repo.
def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

class ProductRecommendationNet(pl.LightningModule):
    def __init__(self,num_products,embedding_size=256,num_heads =4,num_layers=4,learning_rate=1e-3) -> None:
        super().__init__()
        self.product_embedding = nn.Embedding(num_products,embedding_size)
        self.learning_rate = learning_rate
        self.num_products = num_products
        self.positional_embedding = SinusoidalPositionalEmbedding(
            embedding_size,
        )
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
        sequence_embedding += self.positional_embedding(
            sequence_embedding[:, :, 0])
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