from metaflow.plugins.cards.card_modules import MetaflowCard,MetaflowCardComponent
# from metaflow.plugins.card_modules import chevron as pt
import os
import json
from .tables import create_table,Table
from .images import Image
from .charts import LineChart

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

class Heading(MetaflowCardComponent):
    def __init__(self,heading) -> None:
        self._heading = heading

    def render(self):
        return """<h1>%s</h1>"""%self._heading

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
        htmlbody = "\n".join([tables,charts,images]+props)
        render_object = dict(
            head_scripts=self.head_scripts,
            css=self.css,
            body_scripts=self.body_scripts,
            taskpathspec=task.pathspec,
            body_html=htmlbody,
        )
        return pt.render(
            RENDER_TEMPLATE,render_object
        )