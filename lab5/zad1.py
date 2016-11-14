import numpy as np
from PIL import Image
import pywt
import os


def read_image(path):
    return np.array(Image.open(path).convert('L'))


def write_image(image_data, path):
    return Image.fromarray(image_data).convert('L').save(path, quality=100)


def resize_to_enable_dwt_tree(image_data, dwt_depth=2):
    divis = 2 ** dwt_depth
    (height, width) = image_data.shape
    horizontal_offset = width % divis
    vertical_offset = height % divis
    return image_data[vertical_offset:, horizontal_offset:]


def prepare_files(directory, dwt_depth=2, resized_dir='original'):
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    for file in filter(lambda f: f.endswith(".jpg"), os.listdir(directory)):
        raw_image_data = read_image(os.path.join(directory, file))
        data = resize_to_enable_dwt_tree(raw_image_data, dwt_depth=dwt_depth)
        write_image(data, os.path.join(resized_dir, file))


def process_idwt(data, wave, depth):
    if depth == 0:
        return data
    else:
        (h, w) = data.shape
        cA = data[:h // 2, :w // 2]
        cV = data[h // 2:, :w // 2]
        cH = data[:h // 2, w // 2:]
        cD = data[h // 2:, w // 2:]

        cA = process_idwt(cA, wave, depth - 1)

        # cH = np.resize(cH, cA.shape)
        # cV = np.resize(cV, cA.shape)
        # cD = np.resize(cD, cA.shape)

        res = pywt.waverec2((cA, (cH, cV, cD)), wave)
        return res


def process_dwt(data, wave, depth):
    if depth == 0:
        return data
    else:
        (cA, (cH, cV, cD)) = pywt.wavedec2(data, wave, level=1)

        new_cA = process_dwt(cA, wave, depth - 1)

        # large_h, large_w = data.shape
        # cA = cA[-(large_h//2):, -(large_w//2):]
        # cH = cH[-(large_h//2):, -(large_w//2):]
        # cV = cV[-(large_h//2):, -(large_w//2):]
        # cD = cD[-(large_h//2):, -(large_w//2):]

        return np.array(np.vstack((
            np.hstack((new_cA, cH)),
            np.hstack((cV, cD))
        )))


def convert(directory, wave, dwt_depth=2, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(directory, wave)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in filter(lambda f: f.endswith(".jpg"), os.listdir(directory)):
        raw_image_data = read_image(os.path.join(directory, file))
        image_data = resize_to_enable_dwt_tree(raw_image_data, dwt_depth=dwt_depth)

        composed_data = process_dwt(image_data, wave, dwt_depth)

        output_file = os.path.join(output_dir, file)
        write_image(composed_data, output_file)


def dwt():
    directory = "data/selfies"
    resized_dir = "data/selfies/original"
    waves = ['db8']
    dwt_depth = 2

    # prepare_files("data/original", dwt_depth=dwt_depth, resized_dir=resized_dir)
    for wave in waves:
        convert(resized_dir, wave, dwt_depth=dwt_depth, output_dir=os.path.join(directory, wave))


def inverse_dwt():
    directory = "data/selfies/db8"
    wave = 'db8'
    dwt_depth = 2

    output_dir = os.path.join(directory, 'reconstructed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in filter(lambda f: f.endswith(".jpg"), os.listdir(directory)):
        raw_image_data = np.array(Image.open(os.path.join(directory, file)).convert('L'))
        image_data = process_idwt(raw_image_data, wave, dwt_depth)
        write_image(image_data, os.path.join(output_dir, file))


if __name__ == '__main__':
    dwt()
    inverse_dwt()

