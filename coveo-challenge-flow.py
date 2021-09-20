# Todo : Add Flow relating to the coveo challenge flow over here.
from metaflow import FlowSpec,step,batch,S3,Parameter,batch,conda,IncludeFile
import os

class CoveoChallengeFlow(FlowSpec):
    """
    TODO : explain what the flows does.  
    """
    num_rows = Parameter('num-rows',default = 1000000,type=int,help='The number of rows from the dataset to use for Training.')

    batch_size = Parameter('batch-size',default = 64,type=int,help='Batch size to use for training the model.')

    browsing_path_parquet = Parameter(
        'browsing-path-parquet',\
        envvar="BROWSING_PATH_PARQUET",\
        default=None,type=str,help='Path to the browsing.parquet file in S3. This file contains browsing related data in the Coveo Challenge'
    )

    sku_path_parquet = Parameter(
        'sku-path-parquet',\
        envvar="SKU_PATH_PARQUET",\
        default=None,type=str,help='Path to the sku_to_content.parquet files in S3; This file contains unique productids.'
    )

    max_epochs = Parameter(
        'max-epochs',\
        envvar="MAX_EPOCHS",\
        default=1,type=int,help='Maximum number of epochs to train model.'
    )

    num_gpus = Parameter(
        'num-gpus',
        envvar="NUM_GPUS",\
        default=0,type=int,help='Number of GPUs to use when training the model.'
    )
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.2')
    @step
    def start(self):
        """
        
        """
        assert self.browsing_path_parquet  is not None and 's3://' in self.browsing_path_parquet
        assert self.sku_path_parquet  is not None and 's3://' in self.sku_path_parquet
        # setup Dataloading and necessary pre-processing of the data
        from process_data import process_browsing_train
        processed_data = process_browsing_train(self.browsing_path_parquet,num_rows=self.num_rows)
        print(f"Dataframe has the following columns {processed_data.columns}")
        # save as parquet onto S3
        with S3(run=self) as s3:
            # save s3 paths in dict
            self.data_paths = dict()
            s3_root = s3._s3root            
            data_path = os.path.join(s3_root, 'browsing_train.parquet')
            processed_data.to_parquet(path=data_path, engine='pyarrow')
            self.train_data_path = data_path

        print(self.train_data_path)
        
        self.next(self.prepare_dataset)
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.2')
    @step
    def prepare_dataset(self):
        """
        Read in raw session data and build a labelled dataset for purchase/abandonment prediction
        """
        from prepare_dataset import prepare_dataset
        
        # This creates a dictionary of train and validation datasets. 
        # Dictionary of the form : {'train': [[prod_id,prod_id,..]],'val':[]}
        # todo : Make this into a hdf5 process. 
        self.dataset = prepare_dataset(training_file=self.train_data_path,
                                       K=self.num_rows)

        self.next(self.train_model)


    @batch(cpu=4,memory=12000,gpu=2,image='valayob/coveo-challenge-flow-image:0.2')
    @step
    def train_model(self):
        # todo : integrate hyper parameter input + setup + search
        # todo : Find a fast and optimal way to play around with 36M browsing events, 8M search events, 66k Products ;
        # todo : Fan this into a foreach if necessary
        from metaflow import current
        import dataloader
        from dataloader import ProductTokenizer
        from prepare_dataset import read_product_ids
        from models import ProductRecommendationNet
        from pytorch_lightning import Trainer
        product_ids = read_product_ids(self.sku_path_parquet)
        self.tokenizer = ProductTokenizer(product_ids)
        train_loader = dataloader.get_dataloader(
            self.dataset['train'],batch_size=self.batch_size,tokenizer=self.tokenizer
        )
        validation_loader = dataloader.get_dataloader(
            self.dataset['valid'],batch_size=self.batch_size,tokenizer=self.tokenizer
        )
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger("logs", name=current.run_id)
        # trainer = Trainer(logger=logger)
        from pytorch_lightning.callbacks import ModelCheckpoint
        model_checkpoint = ModelCheckpoint(filename='model/checkpoints/{epoch:02d}-{val_loss:.2f}',
                                   save_weights_only=True,
                                   save_top_k=3,
                                   monitor='validation_loss')
        trainer = Trainer(
            max_epochs=self.max_epochs,\
            progress_bar_refresh_rate=25,\
            logger=logger,
            gpus=self.num_gpus,
            callbacks=[model_checkpoint]
        )
        
        model = ProductRecommendationNet(
            len(product_ids),
        )
        trainer.fit(model,train_loader,validation_loader)
        print(f"Best Model Path : {model_checkpoint.best_model_path}")
        trainer.save_checkpoint('last_saved_model.pt')
        with S3(run=self) as s3:
            saved_paths = s3.put_files([\
                ('last_saved_model.pt','./last_saved_model.pt'),\
                ('best_model.pt',model_checkpoint.best_model_path)
            ])
            _,s3_last_url = saved_paths[0]
            _,s3_best_url = saved_paths[1]
            self.last_model_checkpoint = s3_last_url
            self.best_model_checkpoint = s3_best_url

        self.next(self.test_model)

    @step
    def test_model(self):
        self.next(self.end)

    @step
    def end(self):
        print("Completed Executing the flow")


if __name__ == '__main__':
    CoveoChallengeFlow()