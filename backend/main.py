from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    # 转成 BytesIO 对象
    image_stream = io.BytesIO(contents)
    
    # 使用 PIL 打开
    image = Image.open(image_stream)
    


    
    



