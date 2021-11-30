from metaflow.cards import (
    LineChartComponent,
    MetaflowCard,
    PageComponent,
    MetaflowCardComponent,
    TaskInfoComponent,
    SectionComponent,
    TitleComponent,
    RENDER_TEMPLATE
)

import json
import random
from metaflow.plugins.cards.card_modules.basic import TableComponent
TIME_FORMAT = "%Y-%m-%d %I:%M:%S %p"

DEFAULT_OPTIONS = dict(
    only_repr = True, 
    compare_all_users = False,
    max_num_runs = 10,
    only_successful = True,
    line_chart_keys = [
        "loss"
    ]
)


def random_color():
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

def collect_runs(flow,compare_all_users=False,only_successful=True,max_runs= 10):
    if compare_all_users:
        from metaflow import namespace
        namespace(None)
    collected_runs = []
    for run in flow.runs():
        if len(collected_runs) == max_runs:
            break
        if only_successful and not run.successful:
            continue
        collected_runs.append(run)
    return collected_runs
    

class CoveoRunSummaryCard(MetaflowCard):
    type = 'coveo_run_summary_card'

    def __init__(self, options=DEFAULT_OPTIONS, components=[], graph=None):
        self._only_repr = True
        self._graph = graph
        options = self._get_options(options)
        self._only_repr = options["only_repr"]
        self._max_runs = options['max_num_runs']
        self._compare_all_users = options['compare_all_users']
        self._only_successful = options['only_successful']
        self._line_chart_keys = options['line_chart_keys']
        self._components = components

    def _get_options(self,top_level_options):
        options = top_level_options
        if top_level_options is None:
            options = {}
        missing_keys = set(DEFAULT_OPTIONS.keys()) - set(options.keys())
        for k in missing_keys:
            options[k] = DEFAULT_OPTIONS[k]
        return options
    
    def _make_line_chart_component(self,run_list):
        # todo : make this more dynamic. 
        run_losses = []
        for run in run_list:
            if 'loss' in run.data:
                run_losses.append((run,run.data.loss))
        def make_chart_datasets(loss,label):
            selected_color = random_color()
            return {
                "type" : "line",
                "label": label,
                "data": loss,
                "borderWidth": "1",
                "borderColor": selected_color,
                "backgroundColor": selected_color,
            }
        
        return LineChartComponent(chart_config=dict(
            data = dict(
            datasets =  [make_chart_datasets(loss,run.pathspec) for run,loss in run_losses],
            labels = list(range(1,len(run_losses[1][1])))
            )
        ))
    
    def _run_info_section(self,run_list):
        run_details = []
        for run in run_list:
            run_info = dict(
                pathspec = run.pathspec,
                user = None,
                started_at = run.created_at.strftime(TIME_FORMAT),
                finished_at = run.finished_at.strftime(TIME_FORMAT),
                metaflow_version = None,
                successful = run.successful,
                python_version = None
                
            )
            for tag in run.tags:
                if 'user' in tag:
                    run_info['user'] = tag.split('user:')[1]
                elif 'metaflow_version' in tag:
                    run_info['metaflow_version'] = tag.split('metaflow_version:')[1]
                elif 'python_version' in tag:
                    run_info['python_version'] = tag.split('python_version:')[1]
            run_details.append(run_info)
        
        return SectionComponent(
            title='Summary Of Runs',
            contents=[
                TableComponent(headers=list(run_details[0].keys()),data=[list(r.values()) for r in run_details])
            ]
        )
    def render(self, task):        
        # Create a run comparison component. 
        flow = task.parent.parent.parent
        filtered_runs = collect_runs(
            flow,
            max_runs = self._max_runs - 1, # As we add current run here too. 
            compare_all_users = self._compare_all_users,
            only_successful = self._only_successful,
        )
        filtered_runs.insert(0,task.parent.parent)

        run_comparison_section = SectionComponent(title='Comparison Across Recent %d runs' % len(filtered_runs),contents=[
            self._make_line_chart_component(filtered_runs)
        ])
        run_page = PageComponent(title = 'Run Comparisons',contents=[
            self._run_info_section(filtered_runs),
            run_comparison_section
        ]).render()
        final_comp = dict(metadata={
            "pathspec":task.parent.parent.pathspec
        },components=[
            TitleComponent(text=task.parent.parent.pathspec).render(),
            run_page
        ])
        pt = self._get_mustache()
        data_dict = dict(task_data=json.dumps(json.dumps(final_comp)))
        return pt.render(RENDER_TEMPLATE, data_dict)
