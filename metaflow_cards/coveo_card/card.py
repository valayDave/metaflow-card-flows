from metaflow.plugins.cards.card_modules import MetaflowCard,MetaflowCardComponent
# from metaflow.plugins.card_modules import chevron as pt
import os
import json
import uuid
from .charts.chartjs import chart_builder,ChartConfig
from .tables import create_table
from .images import create_image,get_base64image

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
class HelloWorldCard(MetaflowCard):

    type = "helloworld"

    def render(self,task):
        return "<html><body><p>Hello World</p></body></html>"
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


class Table(MetaflowCardComponent):
    def __init__(self,heading,list_of_tuples=[],) -> None:
        self._heading =heading
        self._list_of_tuples = list_of_tuples
    
    def render(self):
        return create_table(self._list_of_tuples,self._heading)

class Image(MetaflowCardComponent):
    def __init__(self,\
                array=None,\
                url=None,\
                caption=None) -> None:
        self._array = array
        self._url = url
        self._caption = caption
    
    def render(self):
        if self._array is not None: 
            path_str = get_base64image(self._array)
        elif self._url is not None:
            path_str = self._url
        return create_image(path_str,'' if self._caption is None else self._caption)


class ModularCard(MetaflowCard):

    type = 'modular_component_card'
    def __init__(self, options=dict(
            # `show_parameters` controls if the card will contain a params table 
            show_parameters=True,
            # These should be links to the Javascipt files
        )):
        main_opts = dict(body_scripts = [CHART_JS_URL],
            # These should be links to the CSS stylesheets
            css = [],
            # These should be links to the head script files
            head_scripts = [])
        main_opts.update(options)
        self._show_parameters,self._body_scripts, \
                self._css , self._head_scripts = self._create_options(main_opts)
        
    
    def _create_options(self,options):
        show_parameters = [] if 'show_parameters' not in options else options['show_parameters']
        body_scripts = [CHART_JS_URL] if 'body_scripts' not in options  else options['body_scripts']
        css = [] if 'css' not in options else options['css']
        head_scripts = [] if 'head_scripts' not in options else options['head_scripts']
        return ( show_parameters, body_scripts, css, head_scripts)

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
        pt = self._get_mustache()
        artifact_ids = []
        for artifact in task:
            artifact_ids.append(artifact.id)
        tables = "" 
        charts = ""
        images = ""
        # Create Tables 
        task_table_data = dict(heading = "Task Metadata",cells=[
            ('Task Created On',task.created_at),
            ('Task Finished At',task.finished_at),
            ('Task Finished',task.finished),
            ('Data Artifacts',', '.join(artifact_ids))
        ])
        tables = create_table(task_table_data['cells'],task_table_data['heading'])
        
        # Fast way to show parameters of a Run
        if self._show_parameters:
            param_ids = [p.id for p in task.parent.parent['_parameters'].task]
            params_table_data = dict(
                cells=[
                    (pid,task[pid].data) for pid in param_ids
                ],
                heading="Flow Parameters"
            )
            tables+='\n'+create_table(params_table_data['cells'],params_table_data['heading'])
        props = []
        if 'card_props' in artifact_ids:
            props = task['card_props'].data

        render_object = dict(
            head_scripts=self.head_scripts,
            css=self.css,
            body_scripts=self.body_scripts,
            taskpathspec=task.pathspec,
            body_html="\n".join([tables,charts,images]+props),
        )
        return pt.render(
            RENDER_TEMPLATE,render_object
        )