from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import sys
from pathlib import Path
import os

# 添加model搜索路径
sys.path.append(str((Path(__file__).parent / "..").resolve() / "model"))
from eval import ImageProcessor

# 读取.env
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
processor = ImageProcessor(os.getenv("CKPT_PATH"))

# 创建fastapi实例
app = FastAPI()


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
    if 1 == len(cam):
        cam = cam[0]

    return {"type": type, "cam": cam}
