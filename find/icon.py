import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_font(size):
    """Get a suitable font, preferring system fonts and falling back to the default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

import math

def create_star_icon(size=40):
    """Generate a red five-pointed star icon on a transparent background at any size."""
    filename = f"star_icon_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    outer_r = size * 0.42
    inner_r = size * 0.18

    points = []
    # Draw the five-pointed star starting from the top.
    start_angle = -math.pi / 2
    for i in range(10):
        angle = start_angle + i * math.pi / 5
        r = outer_r if i % 2 == 0 else inner_r
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))

    draw.polygon(points, fill=(255, 0, 0, 255))

    img.save(output_path)
    print(f"✅ Red five-pointed star icon: {output_path}")

def create_blue_circle_ok(size=40):
    """Generate a blue circle icon with the text 'OK'."""
    filename = f"circle_ok_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    center = size // 2
    radius = int(size * 0.4)
    draw.ellipse([center-radius, center-radius, center+radius, center+radius], fill=(255, 0, 0))
    text = "OK"
    font_size = int(size * 0.35)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    img.save(output_path)
    print(f"✅ Blue circle OK icon: {output_path}")

def create_clock_with_text(width=60, height=40):
    """Generate a clock icon (round face pointing to 3 o'clock) with horizontally arranged text."""
    filename = f"clock_text_{width}x{height}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Clock icon area on the left (about 45% of the width).
    icon_w = int(width * 0.45)
    icon_x = 3
    icon_y = (height - icon_w) // 2
    cx = icon_x + icon_w // 2
    cy = icon_y + icon_w // 2
    r = icon_w // 2 - 1  # Clock-face radius.

    # Draw the round clock face.
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(0, 0, 0), width=1, fill=(255, 255, 255))

    # Draw the hands: hour hand points to 3 and minute hand points to 12.
    hour_length = r - 3
    draw.line([(cx, cy), (cx + hour_length, cy)], fill=(0, 0, 0), width=2)
    minute_length = r - 2
    draw.line([(cx, cy), (cx, cy - minute_length)], fill=(0, 0, 0), width=1)

    # Draw the text "clock" on the right.
    text = "clock"
    max_text_width = width - (icon_x + icon_w + 5)
    font_size = max(8, int(height * 0.35))
    font = get_font(font_size)
    # Dynamically reduce the font size to fit the available width.
    while True:
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_text_width or font_size <= 6:
            break
        font_size -= 1
        font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = icon_x + icon_w + 4
    text_y = (height - text_h) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    img.save(output_path)
    print(f"✅ Clock icon with text ({width}x{height}): {output_path}")

if __name__ == "__main__":
    create_star_icon(40)
    create_blue_circle_ok(40)
    create_clock_with_text(60, 40)
    print(f"\nAll icons have been saved to: {OUTPUT_DIR}")
