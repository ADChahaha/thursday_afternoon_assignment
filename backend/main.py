from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import sys
from pathlib import Path
import os
import base64

# 添加model搜索路径
sys.path.append(str((Path(__file__).parent / "..").resolve() / "model"))
from eval import ImageProcessor

# 读取.env
from dotenv import load_dotenv

base_path = Path(__file__).parent.parent
load_dotenv(base_path / ".env")
processor = ImageProcessor(os.path.join(base_path, os.getenv("CKPT_PATH")))

# 创建fastapi实例
app = FastAPI()

def pil_to_base64(img: Image.Image) -> str:
    """PIL.Image -> base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    """_summary_

    Args:
        file (UploadFile, optional): Assume to be an image with width and length > 0

    Returns:
        return: {"type": , "cam": }
    """
    contents = await file.read()
    # 转成 BytesIO 对象
    image_stream = io.BytesIO(contents)

    # 使用 PIL 打开
    image = Image.open(image_stream)

    # 进行处理
    processor.set_image(image)
    type = processor.eval()
    cam = processor.cam()
    # 对于只有一个图片的情况，直接返回type[0]
    if 1 == len(type):
        type = type[0]
    if len(cam) == 1:
        cam = pil_to_base64(cam[0])
    else:
        cam = [pil_to_base64(c) for c in cam]
    return {"type": type, "cam": cam}
