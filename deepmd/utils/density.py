# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Iterator,
    Tuple,
)

import numpy as np
from scipy.special import logsumexp

class DensityCalculator:
    def __init__(self, filename):
        self.gammaonly = False
        self.ngm_g = 0
        self.nspin = 0
        self.bmat = np.zeros((3, 3))
        self.miller_indices = None
        self.rhog = None
        self.cell_volume = None
        self.read_binary_file(filename)

    def read_binary_file(self, filename: str):
        with open(filename, 'rb') as f:
            # Read header
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == 3, f"Unexpected header value: {size}"
            
            self.gammaonly, self.ngm_g, self.nspin = np.fromfile(f, dtype=np.int32, count=3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the last '3'

            # Read reciprocal lattice vectors
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == 9, f"Unexpected header value: {size}"
            self.bmat = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the last '9'

            # Read Miller indices
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == self.ngm_g * 3, f"Unexpected Miller indices size: {size}"
            self.miller_indices = np.fromfile(f, dtype=np.int32, count=self.ngm_g*3).reshape(self.ngm_g, 3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

            # Read rhog
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == self.ngm_g, f"Unexpected rhog size: {size}"
            self.rhog = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

            # If nspin == 2, read second spin component
            if self.nspin == 2:
                size = np.fromfile(f, dtype=np.int32, count=1)[0]
                assert size == self.ngm_g, f"Unexpected rhog_spin2 size: {size}"
                self.rhog_spin2 = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)
                _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

        # Calculate cell volume
        self.cell_volume = np.abs(np.linalg.det(self.bmat))
        
        # Calculate G vectors
        self.g_vectors = 2 * np.pi * np.dot(self.miller_indices, self.bmat.T)
    
    def print_read_results(self):
        print("Python Read Results:")
        print(f"gamma_only: {self.gammaonly}")
        print(f"ngm_g: {self.ngm_g}")
        print(f"nspin: {self.nspin}")
        
        print("Reciprocal lattice vectors:")
        for i in range(3):
            print(f"  b{i+1}: {self.bmat[i,0]:.6f} {self.bmat[i,1]:.6f} {self.bmat[i,2]:.6f}")
        
        print("First 10 Miller indices:")
        for i in range(min(10, self.ngm_g)):
            print(f"  {self.miller_indices[i,0]} {self.miller_indices[i,1]} {self.miller_indices[i,2]}")
        
        print("First 10 rhog values:")
        for i in range(min(10, self.ngm_g)):
            print(f"  {self.rhog[i].real:.6e} + {self.rhog[i].imag:.6e}i")
        
        if self.nspin == 2:
            print("First 10 rhog_spin2 values:")
            for i in range(min(10, self.ngm_g)):
                print(f"  {self.rhog_spin2[i].real:.6e} + {self.rhog_spin2[i].imag:.6e}i")
    def output_to_files(self):
        # Output Miller indices to file
        np.savetxt('miller_indices_python.txt', self.miller_indices, fmt='%d')

        # Output rhog to file
        with open('rhog_python.txt', 'w') as f:
            f.write("Spin 0:\n")
            for value in self.rhog:
                f.write(f"{value.real} {value.imag}\n")
            
            if self.nspin == 2:
                f.write("Spin 1:\n")
                for value in self.rhog_spin2:
                    f.write(f"{value.real} {value.imag}\n")

        print("Python data has been written to miller_indices_python.txt and rhog_python.txt")

    def calculate_density_batch(self, points: np.ndarray) -> np.ndarray:
        phases = np.exp(1j * np.dot(points, self.g_vectors.T))
        densities = np.real(np.dot(phases, self.rhog)) / self.cell_volume
        return densities
    

def generate_grid(
    lattice_vectors: np.ndarray,
    grid_size: Tuple[int, int, int],
    origin: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """
    生成给定晶格和网格大小的实空间网格点。

    参数:
    lattice_vectors (np.ndarray): 形状为 (3, 3) 的晶格向量
    grid_size (Tuple[int, int, int]): 三个方向上的网格点数
    origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)

    返回:
    np.ndarray: 形状为 (N, 3) 的网格点坐标数组
    """
    nx, ny, nz = grid_size
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)

    grid = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack(grid, axis=-1).reshape(-1, 3)

    # 将分数坐标转换为实空间坐标
    real_points = np.dot(points, lattice_vectors) + origin

    return real_points


def calculate_density(
    filename: str,
    lattice_vectors: np.ndarray,
    grid_size: Tuple[int, int, int],
    origin: np.ndarray = np.zeros(3),
    batch_size: int = 1000,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    计算给定网格的电子密度，以迭代器形式返回结果。

    参数:
    filename (str): 包含 n(G) 数据的二进制文件路径
    grid_size (Tuple[int, int, int]): 三个方向上的网格点数
    origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)
    lattice_vectors (np.ndarray): 形状为 (3, 3) 的晶格向量
    batch_size (int): 每批处理的点数

    返回:
    Iterator[Tuple[np.ndarray, np.ndarray]]: yielding (坐标, 密度值) 对
    """
    calculator = DensityCalculator(filename)

    points = generate_grid(lattice_vectors, grid_size, origin)

    for i in range(0, len(points), batch_size):
        batch_points = points[i : i + batch_size]
        batch_densities = calculator.calculate_density_batch(batch_points)
        yield batch_points, batch_densities
