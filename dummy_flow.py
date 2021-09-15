from metaflow import FlowSpec,step,batch,card

class DummyCardFlow(FlowSpec):
    
    @step
    def start(self):
        self.x = 1
        self.next(self.process)

    @card(type='basic',id='testcard')
    @step
    def process(self):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    DummyCardFlow()