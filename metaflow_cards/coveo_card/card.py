from metaflow.plugins.card_modules.card import MetaflowCard
from metaflow.plugins.card_modules import chevron as pt
import os
import json
from .charts.chartjs import chart_builder,ChartConfig
from .tables import create_table

ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RENDER_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH,'base.html')
CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js"
RENDER_TEMPLATE = None
with open(RENDER_TEMPLATE_PATH,'r') as f:
    RENDER_TEMPLATE = f.read()

def make_script(url):
    return '<script src="%s"></script>' % url

def make_stylesheet(url):
    return '<link href="%s" rel="stylesheet">' % url

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
        "caption":"This is a dummy chart", # Caption of the chart
        "x_key" : "chart_1",  # The key to match in Task object
        "y_key" : "chart_2",  # The key to match in Task object
        "xlabel": "some x label",
        "ylabel": "some x label",
        "chart_type":"line",
        'id' : "cid1"
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

class CoveoDataProcessingCard(MetaflowCard):

    type = 'coveo_data_card'

    def __init__(self,\
                # todo : Give better name to `properties`
                table_cells=[],\
                table_heading = "Task Metadata",
                charts=[],\
                images=[],
                # These should be links to the Javascipt files
                body_scripts = [CHART_JS_URL],
                # These should be links to the CSS stylesheets
                css = [],
                # These should be links to the head script files
                head_scripts = []
                ):
        super().__init__()
        self._charts = charts 
        self._table_cells = table_cells
        self._table_heading = table_heading
        self._images = images
        # self._body_scripts and self._body_css will be 
        self._body_scripts = body_scripts
        self._css = css 
        self._head_scripts = head_scripts
    
    def _make_chart_option(self,task,chart):
        x_data = task[chart['x_key']].data
        y_data = task[chart['y_key']].data
        data_object = dict(
            datasets =  [{
                "label": chart["xlabel"],
                "data": x_data,
                "backgroundColor": "rgb(255, 99, 132)",
                "borderColor": "rgb(255, 99, 132)",
                "borderWidth": "1"
            }],
            labels = y_data
        )

        chart_options = {
        "plugins": {
                "title": {
                    "display": True,
                    "text": chart["caption"]
                }
            },
            "scales": {
                    "y": {
                        "title": {
                            "display":True,
                            "text": chart["ylabel"]
                        }
                    },
                    "x": {
                        "title": {
                            "display":True,
                            "text": chart["xlabel"]
                        }
                }
            }
        }
        return ChartConfig(
            chart_id=chart["id"],\
            data_object=data_object,\
            options=chart_options,\
            chart_type=chart['chart_type']
        )
    
    @property
    def body_scripts(self):
        return "\n".join([make_script(script) for script in self._body_scripts])
    
    @property
    def head_scripts(self):
        return "\n".join([make_script(script) for script in self._head_scripts])
    
    @property
    def css(self):
        return '\n'.join([make_stylesheet(script) for script in self._css])


    def render(self, task):
        # todo : append any images to the body and create a template for that
        artifact_ids = []
        for artifact in task:
            artifact_ids.append(artifact.id)
        tables = "" 
        charts = ""
        images = ""
        if len(self._table_cells) > 0:
            available_cells = []
            for prop in self._table_cells:
                if prop['key'] in artifact_ids:
                    available_cells.append(
                        (prop['name'],task[prop['key']].data)
                    )
            if len(available_cells) > 0:
                tables = create_table(available_cells,self._table_heading)
            
        # check for charts 
        if len(self._charts) > 0 :
            chart_configs = []
            for chart in self._charts:
                # todo : dirty code. Fix later. 
                if chart['x_key'] in artifact_ids and chart['y_key'] in artifact_ids:
                    chart_configs.append(self._make_chart_option(task,chart))
             
            charts = chart_builder(chart_configs)
        
        render_object = dict(
            head_scripts=self.head_scripts,
            css=self.css,
            body_scripts=self.body_scripts,
            taskpathspec=task.pathspec,
            body_html="\n".join([tables,charts,images]),
        )
        return pt.render(
            RENDER_TEMPLATE,render_object
        )
