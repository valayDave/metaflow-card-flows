# Todo : Add Flow relating to the coveo challenge flow over here.
from metaflow import FlowSpec,step,batch

class CoveoChallengeFlow(FlowSpec):

    @step
    def start(self):
        # todo : setup Dataloading and necessary pre-processing of the data
        # todo : setup Hyperparameters
        # todo : Find a fast and optimal way to play around with 36M browsing events, 8M search events, 66k Products ;
        # todo : Fan this into a foreach if necessary
        self.next(self.train_model)

    @step
    def train_model(self):
        self.next(self.test_model)

    @step
    def test_model(self):
        self.next(self.end)

    @step
    def end(self):
        print("Completed Executing the flow")


if __name__ == '__main__':
    CoveoChallengeFlow()