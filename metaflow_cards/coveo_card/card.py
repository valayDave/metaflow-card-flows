from metaflow.cards import (
    LineChartComponent,
    MetaflowCard,
    MetaflowCardComponent,
    TaskInfoComponent,
    TitleComponent,
    RENDER_TEMPLATE
)
import json

class CoveoRunSummaryCard(MetaflowCard):
    type = 'coveo_run_summary_card'

    def __init__(self, options={"only_repr":True}, components=[], graph=None):
        self._only_repr = True
        self._graph = graph
        if "only_repr" in options:
            self._only_repr = options["only_repr"]
        self._components = components
        
    
    def render(self, task):
        task_info_comp = TaskInfoComponent(task,\
                        page_title='Coveo Data Challenge Run Summary',\
                        graph=self._graph,\
                        components=self._components)
        task_info_comp.render()
        page_component = task_info_comp.page_component
        final_comp = task_info_comp.final_component
        final_comp["metadata"]['pathspec'] = task.parent.parent.pathspec
        final_comp["components"] = [
            TitleComponent(text=final_comp["metadata"]['pathspec']).render(),
            page_component
        ]
        pt = self._get_mustache()
        data_dict = dict(task_data=json.dumps(json.dumps(final_comp)))
        return pt.render(RENDER_TEMPLATE, data_dict)
