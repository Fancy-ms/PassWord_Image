import io
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from PIL import Image, ImageOps

app = FastAPI(title="AI Document Photo Generator")
templates = Jinja2Templates(directory="templates")

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

DIMENSIONS = {
    "us": {"w": 600, "h": 600, "ratio": 1.0, "v_padding": 0.52},
    "india": {"w": 413, "h": 413, "ratio": 1.0, "v_padding": 0.52},
    "traffic": {"w": 413, "h": 531, "ratio": 0.7777777777777778, "v_padding": 0.46},
    "student": {"w": 354, "h": 472, "ratio": 0.75, "v_padding": 0.46},
    "company": {"w": 600, "h": 900, "ratio": 0.6666666666666666, "v_padding": 0.38}
}

COLOR_MAP = {
    "white": (255, 255, 255, 255),
    "blue": (0, 102, 204, 255),
    "red": (204, 0, 0, 255)
}

def apply_studio_polish(cv2_bgr_img):
    result = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    balanced = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def intelligent_seam_stretch(pil_img: Image.Image, target_w: int, target_h: int, face_box_crop: tuple) -> Image.Image:
    fx, fy, fw, fh = face_box_crop
    img_np = np.array(pil_img)
    orig_h, orig_w, _ = img_np.shape
    current_ratio = orig_w / orig_h
    target_ratio = target_w / target_h

    if abs(current_ratio - target_ratio) < 0.01:
        return pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    if current_ratio > target_ratio:
        scale_factor = target_h / orig_h
        inter_w = int(orig_w * scale_factor)
        resized_np = cv2.resize(img_np, (inter_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        rx = int(fx * scale_factor)
        crop_diff = inter_w - target_w
        left_cut = max(0, min(int(crop_diff * (rx / (inter_w - int(fw * scale_factor) + 1e-5))), rx))
        return Image.fromarray(resized_np[:, left_cut:left_cut + target_w])
    else:
        needed_h = int(target_w / current_ratio)
        base_resized = cv2.resize(img_np, (target_w, needed_h), interpolation=cv2.INTER_LANCZOS4)
        ry = int(fy * (needed_h / orig_h))
        rh = int(fh * (needed_h / orig_h))
        extra_h = target_h - needed_h

        top_pool, bottom_pool = ry, needed_h - (ry + rh)
        total_pool = top_pool + bottom_pool
        top_add = int(extra_h * (top_pool / total_pool)) if total_pool > 0 else extra_h // 2
        bottom_add = extra_h - top_add

        top_zone = cv2.resize(base_resized[0:ry, :], (target_w, ry + top_add), interpolation=cv2.INTER_LINEAR) if ry > 0 and top_add > 0 else base_resized[0:ry, :]
        bottom_zone = cv2.resize(base_resized[ry+rh:, :], (target_w, bottom_pool + bottom_add), interpolation=cv2.INTER_LINEAR) if bottom_pool > 0 and bottom_add > 0 else base_resized[ry+rh:, :]

        canvas = np.vstack([top_zone, base_resized[ry:ry+rh, :], bottom_zone])
        return Image.fromarray(cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) if canvas.shape[0] != target_h else canvas)

def process_document_sheet(input_bytes: bytes, bg_color_name: str, photo_type: str) -> io.BytesIO:
    import rembg  # Lazy import inside the function so Uvicorn boots instantly

    raw_img = Image.open(io.BytesIO(input_bytes))
    oriented_image = ImageOps.exif_transpose(raw_img)

    # Apply automatic lighting polish
    bgr_cv = cv2.cvtColor(np.array(oriented_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    polished_cv = apply_studio_polish(bgr_cv)
    polished_rgb = cv2.cvtColor(polished_cv, cv2.COLOR_BGR2RGB)
    oriented_image = Image.fromarray(polished_rgb)

    transparent_image = rembg.remove(oriented_image)

    numpy_array = np.array(transparent_image)
    opencv_grayscale = cv2.cvtColor(cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(opencv_grayscale, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

    if len(detected_faces) == 0:
        raise HTTPException(status_code=400, detail="Face detection lost.")

    x, y, width, height = max(detected_faces, key=lambda f: f[2] * f[3])
    face_center_x, face_center_y = x + (width // 2), y + (height // 2)

    spec = DIMENSIONS.get(photo_type, DIMENSIONS["us"])
    target_w, target_h, target_ratio = spec["w"], spec["h"], spec["ratio"]

    box_height = int(height * (2.7 if target_h > target_w else 2.2))
    box_width = int(box_height * target_ratio)

    img_w, img_h = oriented_image.width, oriented_image.height
    while box_width > img_w or box_height > img_h:
        box_height = int(box_height * 0.95)
        box_width = int(box_height * target_ratio)

    if box_height < height * 1.4:
        raise HTTPException(status_code=400, detail="Face alignment error. Step back slightly and retry camera snapshot framing.")

    left = max(0, min(face_center_x - (box_width // 2), img_w - box_width))
    top = max(0, min(face_center_y - int(box_height * spec["v_padding"]), img_h - box_height))

    cropped_face_layer = transparent_image.crop((left, top, left + box_width, top + box_height))
    resized_face = intelligent_seam_stretch(cropped_face_layer, target_w, target_h, (max(0, x - left), max(0, y - top), width, height))

    single_card = Image.new("RGBA", (target_w, target_h), COLOR_MAP.get(bg_color_name, (255, 255, 255, 255)))
    single_card.paste(resized_face, (0, 0), resized_face)
    final_photo = single_card.convert("RGB")

    sheet_w, sheet_h = 1800, 1200
    printable_sheet = Image.new("RGB", (sheet_w, sheet_h), "#F5F5F5")
    cols, rows = sheet_w // (target_w + 40), sheet_h // (target_h + 40)
    start_x, start_y = (sheet_w - (cols * (target_w + 40) - 40)) // 2, (sheet_h - (rows * (target_h + 40) - 40)) // 2

    for r in range(min(rows, 3)):
        for c in range(min(cols, 4)):
            px, py = start_x + c * (target_w + 40), start_y + r * (target_h + 40)
            b_box = Image.new("RGB", (target_w + 2, target_h + 2), "#D3D3D3")
            b_box.paste(final_photo, (1, 1))
            printable_sheet.paste(b_box, (px - 1, py - 1))

    out = io.BytesIO()
    printable_sheet.save(out, format="JPEG", quality=95)
    out.seek(0)
    return out

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"image_generated": False})

@app.post("/generate-passport/", response_class=HTMLResponse)
async def generate_passport(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    bg_color: str = Form("white"),
    photo_type: str = Form("us")
):
    try:
        if image_base64 and image_base64.strip():
            header, encoded = image_base64.split(",", 1)
            file_bytes = base64.b64decode(encoded)
        elif file and file.filename:
            file_bytes = await file.read()
        else:
            raise HTTPException(status_code=400, detail="No source asset provided.")

        raw_image = Image.open(io.BytesIO(file_bytes))
        oriented_image = ImageOps.exif_transpose(raw_image)
        check_gray = cv2.cvtColor(cv2.cvtColor(np.array(oriented_image.convert("RGB")), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(check_gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

        if len(detected_faces) == 0:
            return templates.TemplateResponse(request, "index.html", {
                "request": request, "image_generated": False,
                "error_message": "No face found. Look directly into the yellow overlay framing box."
            })

        processed_sheet = process_document_sheet(file_bytes, bg_color, photo_type)
        encoded_string = base64.b64encode(processed_sheet.getvalue()).decode("utf-8")

        return templates.TemplateResponse(request, "index.html", {
            "request": request, "image_generated": True, "image_data": encoded_string
        })

    except HTTPException as http_err:
        return templates.TemplateResponse(request, "index.html", {"request": request, "image_generated": False, "error_message": http_err.detail})
    except Exception as general_err:
        return templates.TemplateResponse(request, "index.html", {"request": request, "image_generated": False, "error_message": f"Processing error: {str(general_err)}"})
