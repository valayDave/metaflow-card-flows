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
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.7')
    @card(type='modular_component_card',\
        id='datasetcard')
    @step
    def start(self):
        """
        
        """
        from metaflow_cards.coveo_card.card import \
                Table
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
        print("written files to s3",self.train_data_path)
        from metaflow import current
        current.card.extend([
            Table(heading="Dataset Preprocessing Metadata",list_of_tuples=[
                ('Train Data Path',self.train_data_path),
                ('Chosen Columns',', '.join(list(processed_data.columns)))
            ])
        ])
        self.next(self.prepare_dataset)
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.7')
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

        self.next(self.gensim_model,self.torch_model)


    @card(type='modular_component_card',\
        id='gensim_train_card')
    @batch(cpu=4,memory=8000,image='valayob/coveo-challenge-flow-image:0.8')
    @step
    def gensim_model(self):
        self.config = {
            "MIN_C":3,
            "SIZE":48,
            "WINDOW":5,
            "ITERATIONS":self.max_epochs,
            "NS_EXPONENT":0.75
        }
        self.model,self.model_loss,self.epochs = self.train_gensim_model()
        self.last_model_checkpoint = None
        self.best_model_checkpoint = None
        self.model_name = 'Gensim Model'
        from metaflow import current
        current.card.extend(self.gensim_summary(self))
        self.next(self.test_model)
    
    
    def gensim_summary(self,task_object):
        from metaflow_cards.coveo_card.card import \
                LineChart,\
                Table,\
                Image
        return [
            Table(heading="Hyper Parameters",list_of_tuples=self.create_headings(task_object.config)),
            Table(heading='Resuts',list_of_tuples=[
                ("Min Loss",min(task_object.model_loss))
            ]),
            LineChart(
                x = task_object.epochs,
                y = task_object.model_loss,
                xlabel='epochs',
                ylabel='loss',
                caption='Loss Plot'
            )
        ]
        
    @card(type='modular_component_card',\
        id='torch_train_card')
    @batch(cpu=4,memory=30000,gpu=2,image='valayob/coveo-challenge-flow-image:0.8')
    @step
    def torch_model(self):
        self.model_name = 'Torch Model'
        self.config = dict(
            embedding_size=256,
            num_heads =4,
            num_layers=4,
            learning_rate=1e-3
        )
        self.model,self.model_loss,self.epochs = self.train_transformer()
        self.model_loss =  self.transform_metrics(self.model_loss)
        from metaflow import current
        current.card.extend(self.torch_summary(self))
        self.next(self.test_model)

    @card(type='modular_component_card',\
        id='final_summary')
    @step
    def test_model(self,inputs):
        from metaflow_cards.coveo_card.card import Heading
        from metaflow import current
        current.card.append(
            Heading("Summary Of Torch Model")
        )
        current.card.extend(self.torch_summary(
            inputs.torch_model
        ))
        current.card.append(
            Heading("Summary Of Gensim Model")
        )
        current.card.extend(self.gensim_summary(
            inputs.gensim_model
        ))
        self.next(self.end)

    @step
    def end(self):
        print("Completed Executing the flow")



    @staticmethod
    def create_headings(table_dict:dict):
        return [(k.replace('_',' ').title(),v) for k,v in table_dict.items()]

    
    def torch_summary(self,task_object):
        from metaflow_cards.coveo_card.card import \
                LineChart,\
                Table,\
                Image
        return [
            Table(heading='Hyper Parameters',
                list_of_tuples=self.create_headings(task_object.config)
            ),
            Table(heading='Resuts',list_of_tuples=[
                ("Best Model Path",task_object.best_model_checkpoint),
                ("Last Model Path",task_object.last_model_checkpoint),
                ("Min Loss",min(task_object.model_loss['train_loss']['step']))
            ]),
            LineChart(
                x = [i for i in range(len(task_object.model_loss['train_loss']['step']))],
                y = task_object.model_loss['train_loss']['step'],
                xlabel='steps',
                ylabel='Train loss',
                caption='Training Loss Plot'
            ),
            LineChart(
                x =[i for i in range(len(task_object.model_loss['validation_loss']['step']))],
                y = task_object.model_loss['validation_loss']['step'],
                xlabel='steps',
                ylabel='Validation loss',
                caption='Validation Loss Plot'
            )
        ]


    
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
        from prepare_dataset import read_product_ids
        product_ids = read_product_ids(self.sku_path_parquet)
        last_saved_model = 'last_saved_model.pt'
        train_epochs = 3
        model,metrics, best_model_checkpoint = train_transformer(
            self.dataset,
            current.run_id,
            product_ids,
            transformer_args=self.config,
            last_checkpoint_name=last_saved_model,
            max_epochs=train_epochs,
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
        
        return model.state_dict(),metrics, [i for i in range(train_epochs)]

    @staticmethod
    def transform_metrics(metrics):
        import itertools
        loss_dict = dict(
            train_loss = dict(step=[],epoch=[]),
            validation_loss = dict(step=[],epoch=[]),
        )
        def add_to_dict(data_dict,loss_type,step_type):
            main_loss = f'{loss_type}_{step_type}'
            if main_loss in data_dict:
                loss_dict[loss_type][step_type].append(
                    data_dict[main_loss]
                )
                return True
            return False

        type_combs = itertools.product(
            ['train_loss','validation_loss'],
            ['step','epoch']
        )
        for metric_dict,combo in itertools.product(metrics,type_combs):
            if add_to_dict(metric_dict,combo[0],combo[1]):
                pass
        return loss_dict
    

if __name__ == '__main__':
    CoveoChallengeFlow()