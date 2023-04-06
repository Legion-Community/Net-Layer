import numpy as np
import argparse
import os

from PIL import Image
from numba import njit, uint8, int16, int32


def load_img(src):
    img = Image.open(src).convert('RGBA')
    return np.asarray(img)


def save_img(image, path):
    img = Image.fromarray(image)
    img.save(path)


def downscale_img(img, pixel_height, pixel_width):
    height = img.shape[0] // pixel_height
    width = img.shape[1] // pixel_width
    result = np.empty((height, width, img.shape[2]), dtype=np.uint8)
    for i in np.arange(height):
        for j in np.arange(width):
            tile = img[
                i * pixel_height:(i + 1) * pixel_height,
                j * pixel_width:(j + 1) * pixel_width
            ]
            result[i, j] = np.mean(tile, axis=(0, 1))
    return result


def all_lines_count(height, width):
    return (width + 1, height + 1)


def nth_pixels_count(pixel_start, pixel_end, nth_pixel):
    return sum((1 if i % nth_pixel == 0 else 0 for i in range(pixel_start, pixel_end + 1)))


def get_special_lines_count(height, width, nth_pixel, y_start, x_start):
    v_lines_count = nth_pixels_count(x_start, x_start + width, nth_pixel) * 2
    h_lines_count = nth_pixels_count(y_start, y_start + height, nth_pixel) * 2
    return (v_lines_count, h_lines_count)


def regular_lines_count(
    all_vertical_lines,
    all_horizontal_lines,
    special_vertical_lines,
    special_horizontal_lines
):
    return (
        all_vertical_lines - special_vertical_lines,
        all_horizontal_lines - special_horizontal_lines
    )


def get_lines_count(height, width, nth_pixel, y_start, x_start):
    all_lines = all_lines_count(height, width)
    special_lines = get_special_lines_count(height, width, nth_pixel,
                                             y_start, x_start)
    regular_lines = regular_lines_count(*all_lines, *special_lines)
    return (*regular_lines, *special_lines)


def get_result_size(
    height,
    width,
    channels,
    pixel_size,
    regular_lines_width,
    regular_vertical_lines_count,
    regular_horizontal_lines_count,
    special_lines_width,
    special_vertical_lines_count,
    special_horizontal_lines_count,
):
    width_add = (regular_vertical_lines_count * regular_lines_width +
                 special_vertical_lines_count * special_lines_width)
    height_add = (regular_horizontal_lines_count * regular_lines_width +
                  special_horizontal_lines_count * special_lines_width)
    return (height * pixel_size + height_add, width * pixel_size + width_add, channels)


@njit()
def fill_pixels_row(
    pixel_row,
    size,
    x_start,
    pixel_size,
    regular_lines_width,
    nth_pixel,
    special_lines_width
):
    row = np.full(size, 20, dtype=np.uint8)
    pixels_count = pixel_row.shape[0]
    prev_special = False
    idx = 0
    for i in np.arange(pixels_count):
        pos = i + x_start
        if prev_special == True:
            idx += special_lines_width
            prev_special = False
        elif pos % nth_pixel == 0:
            idx += special_lines_width
            prev_special = True
        else:
            idx += regular_lines_width
        row[:,idx:idx+pixel_size] = pixel_row[i]
        idx += pixel_size
    return row


@njit()
def create_line(shape, color):
    line = np.empty(shape, dtype=np.uint8)
    for i in np.arange(shape[0]):
        for j in np.arange(shape[1]):
            line[i, j] = color
    return line


@njit()
def create_templates(
    img,
    size,
    y_start,
    x_start,
    pixel_size,
    regular_lines_width,
    regular_line_color,
    nth_pixel,
    special_lines_width,
    special_line_color
):
    pixel_template = np.full(size, 20, dtype=np.uint8)
    reg_lines_template = np.full(size, -1, dtype=np.int16)
    spec_lines_template = np.full(size, -1, dtype=np.int16)
    regular_vertical_line = create_line((size[0], regular_lines_width, size[2]), regular_line_color)
    regular_horizontal_line = create_line((regular_lines_width, size[1], size[2]), regular_line_color)
    special_vertical_line = create_line((size[0], special_lines_width, size[2]), special_line_color)
    special_horizontal_line = create_line((special_lines_width, size[1], size[2]), special_line_color)
    height, width, channels = img.shape
    row_size = (pixel_size, size[1], channels)
    prev_special = False
    idy = 0
    for i in np.arange(height):
        pos = i + y_start
        if prev_special == True:
            idy += special_lines_width
            prev_special = False
        elif pos % nth_pixel == 0:
            idy += special_lines_width
            prev_special = True
        else:
            idy += regular_lines_width
        row = fill_pixels_row(
            img[i],
            row_size,
            x_start,
            pixel_size,
            regular_lines_width,
            nth_pixel,
            special_lines_width
        )
        pixel_template[idy: idy + pixel_size] = row
        if pos % nth_pixel == 0:
            spec_lines_template[idy - special_lines_width: idy] = special_horizontal_line
            spec_lines_template[idy + pixel_size: idy + pixel_size + special_lines_width] = special_horizontal_line
        else:
            reg_lines_template[idy - regular_lines_width: idy] = regular_horizontal_line
            reg_lines_template[idy + pixel_size: idy + pixel_size + regular_lines_width] = regular_horizontal_line
        idy += pixel_size

    idx = 0
    prev_special = False
    for i in np.arange(width):
        pos = i + x_start
        if prev_special == True:
            idx += special_lines_width
            prev_special = False
        elif pos % nth_pixel == 0:
            idx += special_lines_width
            prev_special = True
        else:
            idx += regular_lines_width
        if pos % nth_pixel == 0:
            spec_lines_template[:, idx - special_lines_width: idx] = special_vertical_line
            spec_lines_template[:, idx + pixel_size: idx + pixel_size + special_lines_width] = special_vertical_line
        else:
            reg_lines_template[:, idx - regular_lines_width: idx] = regular_vertical_line
            reg_lines_template[:, idx + pixel_size: idx + pixel_size + regular_lines_width] = regular_vertical_line
        idx += pixel_size

    return (pixel_template, reg_lines_template, spec_lines_template)


@njit()
def merge_templates(pixel_template, reg_lines_template, spec_lines_template):
    result = np.copy(pixel_template)
    for i in np.arange(result.shape[0]):
        for j in np.arange(result.shape[1]):
            if spec_lines_template[i, j, 0] != -1:
                result[i, j] = spec_lines_template[i, j]
            elif reg_lines_template[i, j, 0] != -1:
                result[i, j] = reg_lines_template[i, j]
    return result


def format_color(channels, color):
    if channels == 4:
        color = (*color, 255)
    color = np.asarray(color, dtype=np.uint8)
    return color


def process_image(
    img,
    pixel_height,
    pixel_width,
    pixel_size,
    y_start,
    x_start,
    regular_line_width,
    regular_color,
    nth_pixel,
    special_line_width,
    special_color
):
    regular_color = format_color(img.shape[2], regular_color)
    special_color = format_color(img.shape[2], special_color)
    if pixel_width != 1 or pixel_height != 1:
        img = downscale_img(img, pixel_height, pixel_width)

    lines_count = get_lines_count(
        img.shape[0],
        img.shape[1],
        nth_pixel,
        y_start,
        x_start
    )

    res_shape = get_result_size(
        *img.shape,
        pixel_size,
        regular_line_width,
        *lines_count[:2],
        special_line_width,
        *lines_count[2:]
    )

    templates = create_templates(
        img,
        res_shape,
        y_start,
        x_start,
        pixel_size,
        regular_line_width,
        regular_color,
        nth_pixel,
        special_line_width,
        special_color
    )

    result = merge_templates(*templates)
    return result


def main(
    input: str,
    output: str,
    pixel_height: int,
    pixel_width: int,
    pixel_size: int,
    line_color: list[int],
    line_width: int,
    nth_pixel: int | None,
    special_line_color: list[int] | None,
    special_line_width: int | None,
    x_start: int,
    y_start: int
):
    if nth_pixel is None:
        nth_pixel = 10
        special_line_color = line_color
        special_line_width = line_width
    if os.path.isfile(input) and os.path.isfile(output):
        image = load_img(input)
        image = process_image(
            image,
            pixel_height,
            pixel_width,
            pixel_size,
            y_start,
            x_start,
            line_width,
            line_color,
            nth_pixel,
            special_line_width,
            special_line_color
        )
        save_img(image, output)
    elif os.path.isdir(input) and os.path.isdir(output):
        for filename in os.listdir(input):
            print(f'Processing file {filename}...')
            input_str = os.path.join(input, filename)
            output_str = os.path.join(output, filename)
            image = load_img(input_str)
            image = process_image(
                image,
                pixel_height,
                pixel_width,
                pixel_size,
                y_start,
                x_start,
                line_width,
                line_color,
                nth_pixel,
                special_line_width,
                special_line_color
            )
            save_img(image, output_str)
    else:
        print('Input and output both must be either file or directory')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility to lay net over pixel image'
    )
    parser.add_argument(
        'input',
        type=str,
        help='path to input image (including extension) or to folder with images'
    )
    parser.add_argument(
        'output',
        type=str,
        help='path to save image (including extension) or to folder where to save images'
    )
    parser.add_argument(
        '-s',
        '--pixel_size',
        default=8,
        type=int,
        help="Final pixel size in pixels"
    )
    parser.add_argument(
        '-ph',
        '--pixel_height',
        default=1,
        type=int,
        help='Height of a single \'pixel\' in real pixels in source'
    )
    parser.add_argument(
        '-pw',
        '--pixel_width',
        default=1,
        type=int,
        help='Width of a single \'pixel\' in real pixels in source'
    )
    parser.add_argument(
        '-c',
        '--line_color',
        type=int,
        nargs=3,
        default=(10, 240, 10),
        help='Color for primary line in rgb format, in range from 0 to 255'
    )
    parser.add_argument(
        '-w',
        '--line_width',
        default=1,
        type=int,
        help='Width of primary line in pixels. Applied after upscaling'
    )
    parser.add_argument(
        '-n',
        '--nth_pixel',
        default=None,
        type=int,
        help='Highlights every n-th pixel with special lines on both axis'
    )
    parser.add_argument(
        '-sc',
        '--special_line_color',
        type=int,
        nargs=3,
        default=(100, 5, 135),
        help='Color for special line in rgb format, in range from 0 to 255. Used when nth_pixel is specified'
    )
    parser.add_argument(
        '-sw',
        '--special_line_width',
        default=1,
        type=int,
        help='Width of special line in pixels. Applied after upscaling. Used when nth_pixel is specified'
    )
    parser.add_argument(
        '-x',
        '--x_start',
        default=0,
        type=int,
        help='Most left coordinate of image in canvas. Count starts from zero. Used for special lines'
    )
    parser.add_argument(
        '-y',
        '--y_start',
        default=0,
        type=int,
        help='Most top coordinate of image in canvas. Count starts from zero. Used for special lines'
    )

    args = parser.parse_args()
    main(**args.__dict__)
