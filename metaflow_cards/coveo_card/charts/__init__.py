from metaflow.plugins.cards.card_modules import MetaflowCardComponent
from .chartjs import chart_builder,ChartConfig
import uuid
class LineChart(MetaflowCardComponent):
    def __init__(self,\
                x=None,\
                y=None,\
                xlabel=None,\
                ylabel=None,\
                caption=None) -> None:
        self._x = x
        self._y = y
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._caption = caption

    def render(self):
        x_data = self._x
        y_data = self._y
        # Making complex datasets here 
        # will require some rethinking about 
        # the exposed `chart` datastructure. 
        data_object = dict(
            datasets =  [{
                "label": self._ylabel,
                "data": self._y,
                "backgroundColor": "rgb(255, 99, 132)",
                "borderColor": "rgb(255, 99, 132)",
                "borderWidth": "1"
            }],
            labels = x_data
        )

        chart_options = {
            "plugins": {
                "title": {
                    "display": True,
                    "text": self._caption
                }
            },
            "scales": {
                    "y": {
                        "title": {
                            "display":True,
                            "text": self._ylabel
                        }
                    },
                    "x": {
                        "title": {
                            "display":True,
                            "text": self._xlabel
                        }
                }
            }
        }
        config = ChartConfig(
            chart_id=str(uuid.uuid4())[:4],\
            data_object=data_object,\
            options=chart_options,\
            chart_type='line'
        )
        return chart_builder([config])

