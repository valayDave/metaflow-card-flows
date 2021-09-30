import os
from metaflow.plugins.card_modules import chevron as pt
ABS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RENDER_TEMPLATE_PATH = os.path.join(ABS_DIR_PATH,'template.html')
RENDER_TEMPLATE = None
with open(RENDER_TEMPLATE_PATH,'r') as f:
    RENDER_TEMPLATE = f.read()

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
