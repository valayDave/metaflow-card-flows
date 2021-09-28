from metaflow import FlowSpec,card,step

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
TABLE_CELLS = [
    {
        "name":"Path of model : (S3 path to the model weights)", # Name is the name of the property to show in the table.
        "key" : "model_wieghts_path"  # The key to match in Task object
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

IMAGES = [
   {
        "caption":"Image From Remote", # Caption for the image.
        # The "key" is the key to match in Task object to retrieve the image. 
        "key" : "",  
        # The artifact Actual path to the image. 
        # This can be a remote path 
        "path_key": "remote_image"
    },
    {
        "caption":"Image From na Artifact",
        "key" : "random_image",
        "path_key": ""
    },
]
class CardPipelineFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.train)

    @card(type='coveo_data_card',\
        options={\
            "charts": CHART_OPTIONS,\
            "table_cells":TABLE_CELLS,\
            "images":IMAGES,\
            "table_heading":"Dummy Table Heading"\
        },\
        id='testcard')
    @step
    def train(self):
        import random
        from metaflow import current
        import numpy as np
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