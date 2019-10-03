import sys
import math

import numpy as np
import matplotlib.pyplot as plt


rcParams = {
    "num_pixels" : 101
}

def rcParam (key, default) :
    return rcParam.get(key, default)

def resolve_rcParam (key, argument, rogue=None) :
    if argument == rogue :
        global rcParams
        return rcParams[key]
    return argument

def generate_ring (**kwargs) :
    num_pixels = resolve_rcParam("num_pixels", kwargs.get("num_pixels", None), None)
    x, y   = np.linspace(0, 1, num_pixels), np.linspace(0, 1, num_pixels)
    x, y   = np.meshgrid(x, y)
    #z      = np.random.normal(30, 5, size=(num_pixels, num_pixels, 1))
    z      = np.zeros(shape=(num_pixels, num_pixels, 1))
    vertex = np.random.uniform(0.2 , 0.8, size=(2,))
    radius = np.random.uniform(0.07, 0.2)
    momentum = 100.*(radius-0.07)/0.13
    cap = 90.
    for idx_x in range(num_pixels) :
        for idx_y in range(num_pixels) :
            res = np.array([x[idx_x, idx_y], y[idx_x, idx_y]]) - vertex
            d   = np.linalg.norm(res) / radius
            d   = 1 - np.fabs(d-1)
            if d < 0 : continue
            z[idx_x, idx_y, 0] = z[idx_x, idx_y, 0] + np.random.normal(130., 5.)*(d**3)
            if z[idx_x, idx_y, 0] > cap : z[idx_x, idx_y, 0] = cap
    if kwargs.get("show", False) :
        plt.imshow(z)
        plt.title(f"{momentum:.0f} GeV at ({vertex[0]:.2f}, {vertex[1]:.2f})")
        plt.show()
    return (vertex[0], vertex[1], momentum, z/cap)

def generate_rings (num, **kwargs) :
    sys.stdout.write("Generating rings...")
    rings = []
    for i in range(num) :
        rings.append(generate_ring(**kwargs))
        if 100*(i+1) % num != 0 : continue
        pct = int(100*(i+1) / num)
        sys.stdout.write(f"\rGenerating rings... {pct}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return rings

def angle_from_A_to_B (A, B) :
    res     = B - A
    len_res = np.linalg.norm(res)
    if len_res == 0 : return 0.
    res   = res / len_res
    angle = np.arccos(np.dot(res, np.array([0, 1])))
    if res[0] > 0 : return angle
    return -1.*angle

def generate_shower (**kwargs) :
    num_pixels = resolve_rcParam("num_pixels", kwargs.get("num_pixels", None), None)
    x, y   = np.linspace(0, 1, num_pixels), np.linspace(0, 1, num_pixels)
    x, y   = np.meshgrid(x, y)
    z      = np.random.normal(30, 5, size=(num_pixels, num_pixels, 1))
    #z      = np.zeros(shape=(num_pixels, num_pixels, 1))
    vertex = np.random.uniform(0.2, 0.8, size=(2,))
    angle  = np.random.uniform(0, np.pi)
    opening_angle = np.random.uniform(.05, .3)
    momentum = 100.*(1.-((opening_angle-0.05)/0.25))
    for idx_x in range(num_pixels) :
        for idx_y in range(num_pixels) :
            point = np.array([x[idx_x, idx_y], y[idx_x, idx_y]])
            theta = angle_from_A_to_B(vertex, point)
            if np.fabs(theta-angle) > opening_angle : continue
            d = np.linalg.norm(point-vertex) / 0.8
            if d > 1 : continue
            d = 1 - d
            z[idx_x, idx_y, 0] = z[idx_x, idx_y, 0] + np.random.normal(40., 10.)*d
    if kwargs.get("show", False) :
        plt.imshow(z)
        plt.title(f"{momentum:.0f} GeV at ({vertex[0]:.2f}, {vertex[1]:.2f})")
        plt.show()
    return (vertex[0], vertex[1], momentum, z/90.)
    
def generate_showers (num, **kwargs) :
    sys.stdout.write("Generating showers...")
    showers = []
    for i in range(num) :
        showers.append(generate_shower(**kwargs))
        if 100*(i+1) % num != 0 : continue
        pct = int(100*(i+1) / num)
        sys.stdout.write(f"\rGenerating showers... {pct}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return showers
