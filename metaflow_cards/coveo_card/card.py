from metaflow.plugins.card_modules.card import MetaflowCard
import os 
from .charts.chartjs import chart_builder,ChartOptions 

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
CHART_JS_PATH = os.path.join(ABS_PATH,'chart.js')

class CoveoDataProcessingCard(MetaflowCard):
    type = 'coveo_data_card'
    def __init__(self) -> None:
        super().__init__()

    def render(self, task):
        mustache = self._get_mustache()
        return mustache.render("%s haha"%task.pathspec)



DEFAULT_PROPERTIES = [
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

DEFAULT_CHARTS = [
    {
        "caption":"", # Caption of the chart
        "key" : "",  # The key to match in Task object
        "xlabel": "",
        "ylabel": ""

    },
]

DEFAULT_IMAGES = [
    {
        "caption":"", # Caption for the image.
        # The "key" is the key to match in Task object to retrieve the image. 
        # This assumes that the image is present in the object. 
        # Ideal expectation is an image
        "key" : "",  
        # Actual path to the image to use for the 
        "path": ""
    },
]

class DummyTestCard(MetaflowCard):
    

    type='dummy_test_card'

    def __init__(self,\
                properties=DEFAULT_PROPERTIES,\
                charts=DEFAULT_CHARTS,\
                images=DEFAULT_IMAGES,
                body_scripts = [CHART_JS_PATH,],
                body_css = [],
                head_css = [],
                head_scripts = []
                ):
        super().__init__()
        self._charts = charts 
        self._properties = properties
        self._images = images
        self._body_scripts = body_scripts
        self._body_css = body_css
        self._head_css = head_css
        self._head_scripts = head_scripts

    def render(self, task):
        pass