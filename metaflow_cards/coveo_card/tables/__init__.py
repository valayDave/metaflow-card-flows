import os
from metaflow.plugins.cards.card_modules import chevron as pt
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
