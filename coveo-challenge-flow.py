# Todo : Add Flow relating to the coveo challenge flow over here.
from metaflow import FlowSpec,step,batch,S3,Parameter,batch,conda
import os

class CoveoChallengeFlow(FlowSpec):
    """
    Take the browsing data and create a new 
    """
    num_rows = Parameter('num-rows',default = 1000000,type=int,help='The number of rows from the dataset to use for Training.')

    browsing_path_parquet = Parameter(
        'browsing-path-parquet',\
        envvar="BROWSING_PATH_PARQUET",\
        default=None,type=str,help='Path to the parquet files in S3 for browsing related data in the Coveo Challenge'
    )

    @step
    def start(self):
        """
        
        """
        assert self.browsing_path_parquet  is not None and 's3://' in self.browsing_path_parquet
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
    
    @step
    def prepare_dataset(self):
        """
        Read in raw session data and build a labelled dataset for purchase/abandonment prediction
        """
        from prepare_dataset import prepare_dataset

        self.dataset = prepare_dataset(training_file=self.train_data_path,
                                       K=self.num_rows)

        self.next(self.train_model)


    @step
    def train_model(self):
        # todo : setup Hyperparameters
        # todo : Find a fast and optimal way to play around with 36M browsing events, 8M search events, 66k Products ;
        # todo : Fan this into a foreach if necessary
        self.next(self.test_model)

    @step
    def test_model(self):
        self.next(self.end)

    @step
    def end(self):
        print("Completed Executing the flow")


if __name__ == '__main__':
    CoveoChallengeFlow()