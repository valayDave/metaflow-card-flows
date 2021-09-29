from metaflow import FlowSpec,card,step,batch,current,Parameter

CHART_OPTIONS = [
    {
        "caption":"This is a first dummy chart", # Caption of the chart
        "x_key" : "x_1",  # The key to match in Task object
        "y_key" : "y_1",  # The key to match in Task object
        "xlabel": "some x label",
        "ylabel": "some y label",
        "chart_type":"line",
        "id" : "cid1",
    },
    {
        "caption":"This is a second dummy chart", # Caption of the chart
        "x_key" : "x_2",  # The key to match in Task object
        "y_key" : "y_2",  # The key to match in Task object
        "xlabel": "some x label",
        "ylabel": "some y label",
        "chart_type":"line",
        "id" : "cid2",
    },
]
TABLES = [
    {
        "heading":"Model Metadata",
        "cells" : [
            {
                "name":"Path of model : (S3 path to the model weights)", # Name is the name of the property to show in the table.
                "key" : "s3_url"  # The key to match in Task object
            },
            {
                "name":"Wandb dashboard URL", 
                "key" :"wandb_url"  
            },
            {
                "name":"Job Execution Meduim", # 
                "key": "exec_medium"
            }
        ]
    },
]

IMAGES = [
   {
        "caption":"Image From Remote URL", # Caption for the image.
        # The "key" is the key to match in Task object to retrieve the image. 
        "key" : "",  
        # The artifact Actual path to the image. 
        # This can be a remote path 
        "path_key": "remote_image"
    },
    {
        "caption":"Image from a Metaflow artifact",
        "key" : "random_image",
        "path_key": ""
    },
]
class CardPipelineFlow(FlowSpec):
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

    @step
    def start(self):
        self.next(self.train)

    @card(type='coveo_data_card',\
        options={\
            "charts": CHART_OPTIONS,\
            "tables": TABLES,\
            "images":IMAGES,\
            "show_parameters":True
        },\
        id='testcard')
    @step
    def train(self):
        import random
        from metaflow import current
        import numpy as np
        self.params = current.parameter_names
        print(self.params)
        self.wandb_url= "<WANDBURL COMES HERE>"
        self.model_wieghts_path= "<MODEL WEIGHTS COMES HERE>"
        self.exec_medium = "local"
        self.random_image = np.random.randn(1024,768).tolist()
        self.remote_image = "https://picsum.photos/1024/768"
        self.y_1 = np.random.randn(10).tolist()
        self.x_1 = [i for i in range(1,10)]
        self.y_2 = [random.randint(0,10) for _ in range(10)]
        self.x_2 = [i for i in range(1,10)]
        
        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")

if __name__ == "__main__":
    CardPipelineFlow()