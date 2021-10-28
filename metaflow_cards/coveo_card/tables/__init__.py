import os
from metaflow.plugins.cards.card_modules import chevron as pt
from metaflow.plugins.cards.card_modules import MetaflowCardComponent
ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RENDER_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH,'template.html')
RENDER_TEMPLATE = None
with open(RENDER_TEMPLATE_PATH,'r') as f:
    RENDER_TEMPLATE = f.read()

def create_table(tuple_list,table_header):
    from tabulate import tabulate
    tablebody = tabulate(tuple_list,tablefmt='html')
    return pt.render(
        RENDER_TEMPLATE,dict(
            tablebody=tablebody,
            tableheader=table_header   
        )
    )


class Table(MetaflowCardComponent):
    def __init__(self,heading,list_of_tuples=[],) -> None:
        self._heading =heading
        self._list_of_tuples = list_of_tuples
    
    def render(self):
        return create_table(self._list_of_tuples,self._heading)
