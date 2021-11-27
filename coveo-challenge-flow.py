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
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.9')
    @card(type='default',\
        id='datasetcard')
    @step
    def start(self):
        """
        
        """
        from metaflow.cards import TableComponent,SectionComponent
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
        current.card.append(
            SectionComponent(
                title="Dataset Preprocessing Metadata",
                contents=[
                TableComponent(
                    headers=[
                        "Train Data Path",
                        "Chosen Columns"
                    ],
                    data=[[
                        self.train_data_path,
                        ', '.join(list(processed_data.columns))
                    ]]
                )],
            )    
        )
        self.next(self.prepare_dataset)
    
    @batch(cpu=4,memory=12000,image='valayob/coveo-challenge-flow-image:0.9')
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

        self.next(self.gensim_model)

    @card(type='default',\
        id='gensim_train_card')
    @batch(cpu=4,memory=8000,image='valayob/coveo-challenge-flow-image:0.9')
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
        self.next(self.end)
    
    
    def gensim_summary(self,task_object):
        from metaflow.cards import \
                LineChartComponent,\
                TableComponent,\
                SectionComponent
        return [
            TableComponent(
                headers=self.create_headings(task_object.config),
                data=self.create_rows(task_object.config)
            ),
            TableComponent(headers='Min Loss',data=[
                [min(task_object.model_loss)]
            ]),
            SectionComponent(title='Loss Plot',subtitle="X-axis is Epochs, Y-Axis is Loss values",contents=[
                LineChartComponent(data=task_object.model_loss,labels=task_object.epochs)
            ])
        ]
        
    # todo : Add a Run Level card over here. 
    @step
    def end(self):
        print("Completed Executing the flow")

    @staticmethod
    def create_headings(table_dict:dict):
        return [(k.replace('_',' ').title()) for k,_ in table_dict.items()]
    
    @staticmethod
    def create_rows(table_dict:dict):
        return [[v for _,v in table_dict.items()]]

    
    def train_gensim_model(self):
        from models.gensim import train_knn
        return train_knn(sessions=self.dataset['train'],
                        min_c=self.config['MIN_C'],
                        size=self.config['SIZE'],
                        window=self.config['WINDOW'],
                        iterations=self.config['ITERATIONS'],
                        ns_exponent=self.config['NS_EXPONENT'])
                        

if __name__ == '__main__':
    CoveoChallengeFlow()