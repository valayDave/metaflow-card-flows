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

class CardPipelineFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.train)

    @card(type='coveo_data_card',options={"charts": CHART_OPTIONS},id='testcard')
    @step
    def train(self):
        import random
        self.y_1 = [random.randint(0,10) for _ in range(10)]
        self.x_1 = [i for i in range(1,10)]
        self.y_2 = [random.randint(0,10) for _ in range(10)]
        self.x_2 = [i for i in range(1,10)]
        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")

if __name__ == "__main__":
    CardPipelineFlow()