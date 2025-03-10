# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.8/doc/doc_ch/quickstart.md
'''

    使用PaddleOCR库进行光学字符识别（OCR），并显示识别结果。具体步骤如下：

'''
from paddleocr import PaddleOCR, draw_ocr
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch",use_gpu=False)  # need to run only once to download and load model into memory
img_path = 'D:/SOD/PaddleSeg-release-2.9.1/tools/ppocr_img/imgs/111.jpg'
result = ocr.ocr(img_path, cls=True)

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[0])

# 显示结果
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('D:/SOD/PaddleSeg-release-2.9.1/tools/ppocr_img/imgs/result.jpg')