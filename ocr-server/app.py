from flask import Flask, request, jsonify
from rapidocr_onnxruntime import RapidOCR
import base64
from PIL import Image
import io
import numpy as np
import os

app = Flask(__name__)
# 全局加载OCR模型（只加载1次，省内存）
ocr_engine = RapidOCR()

@app.route("/ocr", methods=["POST"])
def ocr_api():
    try:
        # 两种传图：表单文件 / json-base64
        if "img" in request.files:
            img_file = request.files["img"]
            img = Image.open(img_file.stream)
        elif request.is_json and "base64_img" in request.json:
            b64 = base64.b64decode(request.json["base64_img"])
            img = Image.open(io.BytesIO(b64))
        else:
            return jsonify({"code":400,"msg":"缺少图片参数"}),400

        img_np = np.array(img)
        res, cost = ocr_engine(img_np)
        text_list = [i[1] for i in res] if res else []
        return jsonify({
            "code":200,
            "cost":round(cost,3),
            "text":text_list,
            "raw":res
        })
    except Exception as e:
        return jsonify({"code":500,"err":str(e)}),500

# Railway自动读取PORT环境变量，必须0.0.0.0
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)