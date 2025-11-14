font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
pil_font = ImageFont.truetype(font_path, 30)

class Common:
    def put_korean_text(img, text, position, font, color=(0, 255, 0)):
    """OpenCV 이미지에 한글 텍스트 추가"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)