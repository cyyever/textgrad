def is_jpeg(data):
    jpeg_signature = b"\xff\xd8\xff"
    return data.startswith(jpeg_signature)


def is_png(data):
    png_signature = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    return data.startswith(png_signature)


def get_image_type_from_bytes(data):
    if is_jpeg(data):
        return "jpeg"
    elif is_png(data):
        return "png"
    else:
        raise ValueError("Image type not supported, only jpeg and png supported.")
