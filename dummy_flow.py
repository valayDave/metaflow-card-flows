from metaflow import FlowSpec,step,batch,current,Parameter,card

class ModelTrainingFlow(FlowSpec):
    num_rows = Parameter('num-rows',default = 1000000,type=int,help='The number of rows from the dataset to use for Training.')

    batch_size = Parameter('batch-size',default = 64,type=int,help='Batch size to use for training the model.')

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

    @step
    def train(self):
        import random
        import numpy as np
        self.exec_medium = "local"
        self.train_result_image,self.model_wieghts,self.logger_url = self.train_model()
        self.loss = (np.random.randn(100)*100).tolist()
        self.next(self.end)
    
    def train_model(self):
        import pandas as pd
        import numpy as np
        import io
        import torch
        import matplotlib.pyplot as plt
        df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))
        plot = df.plot()
        fig = plot.get_figure()
        fig.show()
        b = io.BytesIO()
        plt.savefig(b, format='png')
        return b.getvalue(),torch.randn(10,10),"<WANDBURL COMES HERE>"
    
    @card(type='default',id='train_card')
    @step
    def end(self):
        from metaflow.cards import SectionComponent,LineChartComponent
        from metaflow import current
        chart_component = SectionComponent(title="Loss Plot",
            contents=[
                LineChartComponent(
                    data=self.loss,
                    labels=list(range(1,len(self.loss)))
                )
        ])
        current.card.append(
            chart_component
        )
        print("Done Computation")

if __name__ == "__main__":
    ModelTrainingFlow()