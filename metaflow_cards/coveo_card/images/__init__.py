import os
from metaflow.plugins.cards.card_modules import chevron as pt
from metaflow.plugins.cards.card_modules import MetaflowCardComponent
ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RENDER_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH,'template.html')
RENDER_TEMPLATE = None
with open(RENDER_TEMPLATE_PATH,'r') as f:
    RENDER_TEMPLATE = f.read()


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


def get_base64image(img_data):
    try:
        from PIL import Image
        import base64
        import numpy as np
        from io import BytesIO
        image = Image.fromarray(np.uint8(img_data))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return 'data:image/png;base64, %s'%img_str.decode('utf-8')
    except ImportError as e:
        # todo : don't swallow the error over here. 
        return None

def create_image(image_src,caption):
    return pt.render(
        RENDER_TEMPLATE,dict(
            imgsrc=image_src,
            caption=caption   
        )
    )
