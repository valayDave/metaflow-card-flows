import os 
from dataclasses import dataclass, asdict
from metaflow.plugins.card_modules import chevron as pt

CURRDIR = os.path.dirname(os.path.abspath(__file__))
# Import the individual chart template. 
SINGLE_CHART_TEMPLATE_PATH = os.path.join(CURRDIR,'template.html')
CHART_SCRIPT_TEMPLATE = None
with open(SINGLE_CHART_TEMPLATE_PATH,'r') as f:
    CHART_SCRIPT_TEMPLATE = f.read()

@dataclass
class ChartConfig:
    chart_id:str = None
    data_object:dict = None
    chart_type:str = None
    options:dict = None
    
    def __post_init__(self):
        assert 'datasets' in self.data_object
        assert 'labels' in self.data_object


def chart_builder(chart_configs = []):
    """
    Args:
        chart_options (List[ChartOption]): 
    """
    # Todo Switch remote JS with minified code of chart.js
    tags = ''
    for config in chart_configs:
        chart_kwags = asdict(config)
        chart = Chart(**chart_kwags)
        tags+=chart.chart_tag + '\n'+ chart.chart_script_tag+'\n'

    return tags
    
    

class Chart:
    def __init__(self,
                chart_id=None,
                data_object={},
                chart_type=None,
                options={}
                ) -> None:
        """
        This class will easily help compile the chart 
        related information for the input data. 
        Args:
            chart_id : Id given to the chart
            data_object : Actual dataset object like the one used in chart.js
                - For reference to what this object shoudl look like check : 
                    - https://www.chartjs.org/docs/latest/api/interfaces/ChartData.html , 
                    - https://www.chartjs.org/docs/latest/api/#chartdataset , 
                - 
            chart_type : type of the given chart. 
        """
        assert 'labels' in data_object
        assert 'datasets' in data_object
        assert len(data_object['datasets']) > 0
        self._id = chart_id
        self._type = chart_type
        self._dataobject = data_object
        self._options =options
    
    @property
    def chart_tag(self):
        return '<div class="row"><div class="col-xs-12"><canvas id="%s"></canvas></div></div>' % (self._id)
    
    @property
    def chart_script_tag(self):
        
        return pt.render(
            CHART_SCRIPT_TEMPLATE,
            data=dict(
                chart_id=self._id,
                data_object=self._dataobject,
                chart_type = self._type,
                chart_options=self._options
            )
        )
