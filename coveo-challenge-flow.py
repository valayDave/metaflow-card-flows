# Todo : Add Flow relating to the coveo challenge flow over here.
from metaflow import FlowSpec, current,step,batch,S3,Parameter,batch,conda,IncludeFile,card
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
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.6')
    @step
    def start(self):
        """
        
        """
        from metaflow import current
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
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.6')
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


    # @batch(cpu=4,memory=8000,image='valayob/coveo-challenge-flow-image:0.6')
    @card(type='coveo_data_card',\
        options={\
            "show_parameters":True
        },\
        id='training_card')
    @step
    def train_model(self):
        self.config = {
            "MIN_C":3,
            "SIZE":48,
            "WINDOW":5,
            "ITERATIONS":15,
            "NS_EXPONENT":0.75
        }
        self.model,self.model_loss = self.train_gensim_model()
        self.next(self.test_model)
        
    
    def train_gensim_model(self):
        from models.gensim import train_knn
        return train_knn(sessions=self.dataset['train'],
                        min_c=self.config['MIN_C'],
                        size=self.config['SIZE'],
                        window=self.config['WINDOW'],
                        iterations=self.config['ITERATIONS'],
                        ns_exponent=self.config['NS_EXPONENT'])
    
    def train_transformer(self):
        # todo : integrate hyper parameter input + setup + search
        # todo : Find a fast and optimal way to play around with 36M browsing events, 8M search events, 66k Products ;
        # todo : Fan this into a foreach if necessary
        from models.transformer import train_transformer
        last_saved_model = 'last_saved_model.pt'
        model,best_model_checkpoint = train_transformer(
            self.dataset,
            self.sku_path_parquet,
            current.run_id,
            last_checkpoint_name=last_saved_model,
            num_gpus=self.num_gpus,
            max_epochs=self.max_epochs,
            batch_size= self.batch_size
        )
        with S3(run=self) as s3:
            saved_paths = s3.put_files([\
                (last_saved_model,f'./{last_saved_model}'),\
                ('best_model.pt',best_model_checkpoint)
            ])
            _,s3_last_url = saved_paths[0]
            _,s3_best_url = saved_paths[1]
            self.last_model_checkpoint = s3_last_url
            self.best_model_checkpoint = s3_best_url
        return model

    @step
    def test_model(self):
        self.next(self.end)

    @step
    def end(self):
        print("Completed Executing the flow")


if __name__ == '__main__':
    CoveoChallengeFlow()