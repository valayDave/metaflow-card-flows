from metaflow import FlowSpec,card,step,batch,current,Parameter

class ModularComponentCardFlow(FlowSpec):
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

    @card(type='modular_component_card',\
        timeout=10,
        id='modcard')
    @step
    def start(self):
        from metaflow_cards.coveo_card.card import \
                LineChart,\
                Table,\
                Image
        import numpy as np
        self.wandb_url= "<WANDBURL COMES HERE>"
        self.model_wieghts_path= "<MODEL WEIGHTS COMES HERE>"
        self.exec_medium = "local"
        self.x1 = [i for i in range(1,10)]
        self.y1 = np.random.randn(10).tolist()
        self.y2 = np.random.randn(10).tolist()
        self.add_to_card([
            Table(heading='My Flows metadata',list_of_tuples=[
                ('Wandb url',self.wandb_url),
                ('Model Weights',self.model_wieghts_path),
                ("Execution Medium",self.exec_medium)
            ]),
            LineChart(x=self.x1,
                    y=self.y1,
                    caption='my loss chart',
                    xlabel='epoch',
                    ylabel='loss'),
            LineChart(x=self.x1,
                    y=self.y2,
                    caption='my loss chart',
                    xlabel='epoch',
                    ylabel='loss'),
            Image(caption="My Random Image From An Array",array=np.random.randn(1024,768).tolist()),

        ])
        self.next(self.end)
    
   
    @step
    def end(self):
        print("Done Computation")

if __name__ == "__main__":
    ModularComponentCardFlow()