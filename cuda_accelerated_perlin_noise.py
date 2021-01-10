import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pyperlin import FractalPerlin2D
from tqdm import tqdm


def map_to_colors(noise, color_steps):
    colors = {
        'deep_water': (45, 85, 205),
        'water': (65, 105, 225),
        'beach': (238, 214, 175),
        'grass': (34, 139, 34),
        'mountain': (139, 137, 137),
        'snow': (255, 250, 250)
    }
    y_len, x_len = noise.shape
    noise_color = np.zeros((y_len, x_len, 3), dtype=np.int32)
    for y in range(y_len):
        for x in range(x_len):
            for step, color in color_steps:
                if noise[y][x] <= step:
                    noise_color[y][x] = colors[color]
                    break
    return noise_color


def map_to_grey(noise):
    y_len, x_len = noise.shape
    noise_color = np.zeros((y_len, x_len, 3), dtype=np.float)
    for y in range(y_len):
        for x in range(x_len):
            noise_color[y][x] = ((noise[y][x] + 1) / 2,) * 3
    return noise_color


def plot_noises(noises):
    fig = plt.figure(figsize=(15, 10))

    subplots = range(231, 237)
    titles = [
        'CPU original', 'CPU 3 layers', 'CPU 6 layers',
        'CUDA original', 'CUDA 3 layers', 'CUDA 6 layers'
    ]

    for subplot, title, noise in zip(subplots, titles, noises):
        ax = fig.add_subplot(subplot)
        ax.set_axis_off()
        ax.set_title(title)
        ax.imshow(noise)

    fig.show()


def generate_noise(device, zoom=1):
    # for batch size = 1 and noises' shape = (1024,1024)
    shape = (1, 512 * zoom, 512 * zoom)

    # generator
    generator = torch.Generator(device=device)

    # for lacunarity = 2.0
    resolutions = [(zoom * 2 ** i, zoom * 2 ** i) for i in range(1, 8)]
    # for persistence = 0.5
    factors = [.5 ** i for i in range(7)]

    # initialize objects
    fp = FractalPerlin2D(shape, resolutions, factors, generator=generator)

    # benchmark Perlin Noise time
    start = time.time()
    noise = fp()
    end = time.time()

    noise = noise.cpu().numpy()[0]
    del fp

    return noise, end - start


def main(zoom):
    noise_cpu, elapsed_time = generate_noise('cpu', zoom)
    print(f'Perlin Noise CPU time: {elapsed_time}')
    noise_cuda, elapsed_time = generate_noise('cuda', zoom)
    print(f'Perlin Noise CUDA time: {elapsed_time}')

    print('Generating images')

    # reduce the number of steps in noise
    color_steps_3_layers = [
        (-1 / 6, 'water'),
        (0, 'beach'),
        (1, 'grass')
    ]
    color_steps_6_layers = [
        (-1 / 2, 'deep_water'),
        (-1 / 6, 'water'),
        (0, 'beach'),
        (1 / 3, 'grass'),
        (1 / 2, 'mountain'),
        (1, 'snow')
    ]
    noise_cpu_grey = map_to_grey(noise_cpu)
    noise_cuda_grey = map_to_grey(noise_cuda)
    noise_cpu_3_layers = map_to_colors(noise_cpu, color_steps_3_layers)
    noise_cuda_3_layers = map_to_colors(noise_cuda, color_steps_3_layers)
    noise_cpu_5_layers = map_to_colors(noise_cpu, color_steps_6_layers)
    noise_cuda_5_layers = map_to_colors(noise_cuda, color_steps_6_layers)
    noises = [
        noise_cpu_grey, noise_cpu_3_layers, noise_cpu_5_layers,
        noise_cuda_grey, noise_cuda_3_layers, noise_cuda_5_layers
    ]
    plot_noises(noises)


def benchmark(number_iterations, zoom):
    print(f'Doing Perlin Noise benchmark with {number_iterations} iterations')
    elapsed_time_cpu_sum = 0
    elapsed_time_cuda_sum = 0
    for _ in tqdm(range(number_iterations)):
        _, elapsed_time_cpu = generate_noise('cpu', zoom)
        _, elapsed_time_cuda = generate_noise('cuda', zoom)
        elapsed_time_cpu_sum += elapsed_time_cpu
        elapsed_time_cuda_sum += elapsed_time_cuda
    elapsed_time_cpu_average = elapsed_time_cpu_sum / number_iterations
    elapsed_time_cuda_average = elapsed_time_cuda_sum / number_iterations
    print(f'Average Perlin Noise CPU time: {elapsed_time_cpu_average}')
    print(f'Average Perlin Noise CUDA time: {elapsed_time_cuda_average}')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate image or benchmark')
    parser.add_argument(
        '-b', '--bench', '--benchmark',
        type=int,
        default=0,
        help='number of iterations (default 0)')
    parser.add_argument(
        '-i', '--image',
        action='store_true',
        help='generate image'
    )
    parser.add_argument(
        '-z', '--zoom',
        type=int,
        default=1,
        help='zoom level (default 1)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.image:
        main(args.zoom)
    if args.bench:
        benchmark(args.bench, args.zoom)
    input('Press ENTER to continue...')
    plt.close()
