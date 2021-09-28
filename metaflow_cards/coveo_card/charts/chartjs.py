import os 
from dataclasses import dataclass, asdict
from metaflow.plugins.card_modules import chevron as pt

CURRDIR = os.path.dirname(os.path.abspath(__file__))
# Import the individual chart template. 
SINGLE_CHART_TEMPLATE_PATH = os.path.join(CURRDIR,'chart_script_compiler.txt')
CHART_SCRIPT_TEMPLATE = None
with open(SINGLE_CHART_TEMPLATE_PATH,'r') as f:
    CHART_SCRIPT_TEMPLATE = f.read()

# Import the main container template where all the charts get pasted. 
MAIN_HTML_TEMPLATE_PATH = os.path.join(CURRDIR,'base_template.html')
MAIN_HTML = None
with open(MAIN_HTML_TEMPLATE_PATH,'r') as f:
    MAIN_HTML = f.read()


@dataclass
class ChartOptions:
    chart_id:str = None
    data_object:dict = None
    chart_type:str = None


def chart_builder(chart_options = []):
    """[summary]

    Args:
        chart_options (list, optional): [description]. Defaults to [].
    """
    tags = ''
    for option in chart_options:
        chart_kwags = asdict(option)
        chart = Chart(**chart_kwags)
        tags+=chart.chart_tag + '\n'+ chart.chart_script_tag+'\n'

    return pt.render(
        MAIN_HTML,
        data = dict(chart_tags = tags)
    )
    
    

class Chart:
    def __init__(self,
                chart_id=None,
                data_object={},
                chart_type=None,
                ) -> None:
        """
        This class will easily help compile the chart 
        related information for the input data. 
        Args:
            chart_id : Id given to the chart
            data_object : Actual dataset object like the one used in chart.js
                - For reference to what this object shoudl look like check : https://www.chartjs.org/docs/latest/api/interfaces/ChartData.html , https://www.chartjs.org/docs/latest/api/#chartdataset , 
                - 
            chart_type : type of the given chart. 
        """
        assert 'labels' in data_object
        assert 'datasets' in data_object
        assert len(data_object['datasets']) > 0
        self._id = chart_id
        self._type = chart_type
        self._dataobject = data_object
    
    @property
    def chart_tag(self):
        return '<div id="%s"><canvas id="%s"></canvas></div>' % (self._id,self._id)
    
    @property
    def chart_script_tag(self):
        
        return pt.render(
            CHART_SCRIPT_TEMPLATE,
            data=dict(
                chart_id=self._id,
                data_object=self._dataobject,
                chart_type = self._type,
            )
        )
