import numpy as np
import argparse

from PIL import Image


def load_img(src):
    img = Image.open(src)
    return np.asarray(img)


def upscale_img(image, pixel_height, pixel_width):
    height, width, channels = image.shape
    r_height = height * pixel_height
    r_width = width * pixel_width
    result = np.empty((r_height, r_width, channels), dtype=image.dtype)
    for i in np.arange(0, height):
        for j in np.arange(0, width):
            result[i * pixel_height:(i + 1) * pixel_height, j * pixel_width:(j + 1) * pixel_width] = image[i, j]
    return result


def create_line(height, width, color, dtype):
    return np.asarray(
        [[color for _ in range(width)] for _ in range(height)],
        dtype=dtype
    )


def add_horizontal_lines(
    image,
    pixel_height,
    line_color,
    line_width,
    nth_line=None,
    nth_line_color=None,
    nth_line_width=None,
    count_start=0
):
    height, width, channels = image.shape
    line = create_line(line_width, width, line_color, image.dtype)
    if nth_line:
        special_line = create_line(nth_line_width, width,
                                   nth_line_color, image.dtype)
    result = np.empty((0, width, channels), dtype=image.dtype)
    idx = count_start
    for i in np.arange(0, height // pixel_height):
        result = np.append(
            result, image[i*pixel_height:(i + 1) * pixel_height], axis=0
        )
        if (i + 1) * pixel_height < height:
            if nth_line and (idx + 1) % nth_line == 0:
                result = np.append(result, special_line, axis=0)
            else:
                result = np.append(result, line, axis=0)
            idx += 1
    return result


def add_vertical_lines(
    image,
    pixel_width,
    line_color,
    line_width,
    nth_line=None,
    nth_line_color=None,
    nth_line_width=None,
    count_start=0
):
    height, width, channels = image.shape
    line = create_line(height, line_width, line_color, image.dtype)
    if nth_line:
        special_line = create_line(height, nth_line_width,
                                   nth_line_color, image.dtype)
    result = np.empty((height, 0, channels), dtype=image.dtype)
    idx = count_start
    for i in np.arange(0, width // pixel_width):
        result = np.append(
            result, image[:, i * pixel_width:(i + 1) * pixel_width], axis=1
        )
        if (i + 1) * pixel_width < width:
            if nth_line and (idx + 1) % nth_line == 0:
                result = np.append(result, special_line, axis=1)
            else:
                result = np.append(result, line, axis=1)
            idx += 1
    return result


def add_lines(
    image,
    pixel_height,
    pixel_width,
    line_color,
    line_width,
    nth_line=None,
    nth_line_color=None,
    nth_line_width=None,
    vertical_count_start=0,
    horizontal_count_start=0
):
    print('Adding horizontal lines...')
    image = add_horizontal_lines(
        image,
        pixel_height,
        line_color,
        line_width,
        nth_line,
        nth_line_color,
        nth_line_width,
        horizontal_count_start
    )
    print('Finished adding horizontal lines')

    print('Adding vertical lines...')
    image = add_vertical_lines(
        image,
        pixel_width,
        line_color,
        line_width,
        nth_line,
        nth_line_color,
        nth_line_width,
        vertical_count_start
    )
    print('Finished adding vertical lines')

    return image


def save_img(image, path):
    img = Image.fromarray(image)
    img.save(path)


def main(
    input: str,
    output: str,
    upscale_x: int,
    upscale_y: int,
    pixel_height: int,
    pixel_width: int,
    line_color: list[int],
    line_width: int,
    nth_line: int | None,
    nth_line_color: list[int] | None,
    nth_line_width: int | None,
    x_start: int,
    y_start: int
):
    img = load_img(input)
    if upscale_x != 1 and upscale_y != 1:
        print('Upscaling image...')
        img = upscale_img(img, upscale_y, upscale_x)
        print('Finished upscaling image')
    r_pixel_h = pixel_height * upscale_y
    r_pixel_w = pixel_width * upscale_x
    if img.shape[2] == 4:
        line_color = (*line_color, 255)
        if nth_line:
            nth_line_color = (*nth_line_color, 255)
    img = add_lines(
        img,
        pixel_height=r_pixel_h,
        pixel_width=r_pixel_w,
        line_color=line_color,
        line_width=line_width,
        nth_line=nth_line,
        nth_line_color=nth_line_color,
        nth_line_width=nth_line_width,
        vertical_count_start=y_start,
        horizontal_count_start=x_start
    )
    print('Saving image...')
    save_img(img, output)
    print(f'Image saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility to lay net over pixel image'
    )
    parser.add_argument('input', type=str,
                        help='path to input image (including extension)')
    parser.add_argument('output', type=str,
                        help='path to save image (including extension)')
    parser.add_argument('-ux', '--upscale_x', default=8, type=int)
    parser.add_argument('-uy', '--upscale_y', default=8, type=int)
    parser.add_argument('-ph', '--pixel_height', default=1, type=int,
                        help='Height of a single \'pixel\' in real pixels')
    parser.add_argument('-pw', '--pixel_width', default=1, type=int,
                        help='Width of a single \'pixel\' in real pixels')
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
        '--nth_line',
        default=None,
        type=int,
        help='Makes every nth line special with other settings. If not specified it won\'t exist'
    )
    parser.add_argument(
        '-nc',
        '--nth_line_color',
        type=int,
        nargs=3,
        default=(200, 0, 200),
        help='Color for special line in rgb format, in range from 0 to 255. Used when nth_line is specified'
    )
    parser.add_argument(
        '-nw',
        '--nth_line_width',
        default=2,
        type=int,
        help='Width of special line in pixels. Applied after upscaling. Used when nth_line is specified'
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
