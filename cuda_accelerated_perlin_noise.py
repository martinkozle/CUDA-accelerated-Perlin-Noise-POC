import time
import argparse
from typing import Iterable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyperlin import FractalPerlin2D
from tqdm import tqdm

colors = {
    'deep_water': (45, 85, 205),
    'water': (65, 105, 225),
    'beach': (238, 214, 175),
    'grass': (34, 139, 34),
    'mountain': (139, 137, 137),
    'snow': (255, 250, 250)
}


def map_to_colors(noise: np.ndarray,
                  color_steps: Iterable[Tuple[float, str]]
                  ) -> np.ndarray:
    y_len, x_len = noise.shape
    noise_color = np.zeros((y_len, x_len, 3), dtype=np.int32)
    for y in range(y_len):
        for x in range(x_len):
            for step, color in color_steps:
                if noise[y][x] <= step:
                    noise_color[y][x] = colors[color]
                    break
    return noise_color


def map_to_grey(noise: np.ndarray) -> np.ndarray:
    y_len, x_len = noise.shape
    noise_color = np.zeros((y_len, x_len, 3), dtype=np.float)
    for y in range(y_len):
        for x in range(x_len):
            noise_color[y][x] = ((noise[y][x] + 1) / 2,) * 3
    return noise_color


def plot_2d_noises(noises: Iterable[np.ndarray]) -> None:
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

    size = len(noises[0])
    plt.savefig(f'images/plot_2d_{size}x{size}.png')
    fig.show()


def plot_3d_noise(noise: np.ndarray, title_prefix: str) -> None:
    terrain_colorscale = [
        (0, f'rgb{colors["deep_water"]}'),
        (0.4, f'rgb{colors["water"]}'),
        (0.5, f'rgb{colors["beach"]}'),
        (0.55, f'rgb{colors["grass"]}'),
        (0.75, f'rgb{colors["mountain"]}'),
        (0.9, f'rgb{colors["snow"]}'),
        (1, f'rgb{colors["snow"]}'),
    ]

    sea_level_colorscale = [
        (0, f'rgb{colors["water"]}'),
        (1, f'rgb{colors["water"]}')
    ]

    size = len(noise)

    terrain = np.array([row[:, 0] * 100 for row in noise[::-1]])
    sea_level = np.array([[45] * size for _ in range(size)])

    terrain_surface = go.Surface(
        z=terrain,
        colorscale=terrain_colorscale,
        cmin=0, cmax=100
    )
    sea_level_surface = go.Surface(
        z=sea_level,
        colorscale=sea_level_colorscale,
        opacity=0.3,
        showscale=False
    )

    fig = go.Figure(data=[terrain_surface, sea_level_surface])
    fig.update_traces(
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        )
    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=-0.2, y=0, z=-0.5),
        eye=dict(x=0.5, y=-2, z=1.25)
    )
    fig.update_layout(
        title=f'{title_prefix} {size}x{size}',
        autosize=False,
        width=1400, height=900,
        margin=dict(l=65, r=50, b=65, t=90),
        scene_aspectmode='data',
        scene_camera=camera
    )
    fig.write_image(f'images/plot_3d_{title_prefix}_{size}x{size}.png')
    fig.show()


def generate_noise(device: str, zoom: int = 1) -> Tuple[np.ndarray, float]:
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


def main(zoom: int, plot2d: bool, plot3d: bool) -> None:
    noise_cpu, elapsed_time = generate_noise('cpu', zoom)
    print(f'Perlin Noise CPU time: {elapsed_time}')
    noise_cuda, elapsed_time = generate_noise('cuda', zoom)
    print(f'Perlin Noise CUDA time: {elapsed_time}')

    print('Generating visualizations')

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

    if plot2d:
        plot_2d_noises(noises)
        input('Showing 2D noises plot. Press ENTER to continue...')
    if plot3d:
        plot_3d_noise(noise_cpu_grey, 'CPU')
        input('Showing 3D CPU noise plot. Press ENTER to continue...')
        plot_3d_noise(noise_cuda_grey, 'CUDA')
        input('Showing 3D CUDA noise plot. Press ENTER to continue...')


def benchmark(number_iterations: int, zoom: int) -> None:
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
        '--p2d', '--plot2d',
        action='store_true',
        help='generate 2d plots'
    )
    parser.add_argument(
        '--p3d', '--plot3d',
        action='store_true',
        help='generate 3d plots',
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
    if args.p2d or args.p3d:
        main(args.zoom, args.p2d, args.p3d)
    if args.bench:
        benchmark(args.bench, args.zoom)
        input('Finished benchmarking. Press ENTER to continue...')
    plt.close()
