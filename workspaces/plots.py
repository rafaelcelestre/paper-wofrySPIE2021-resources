#!/bin/python


import sys
sys.path.insert(0, '../srw_python')
from srwlib import *

import numpy as np

import barc4plots.barc4plots as b4pt

def read_srw_intensity_dat(file_name, bandwidth=1e-3, transmission=1, norm=False):

    image, mesh = srwl_uti_read_intens_ascii(file_name)
    image = np.reshape(image, (mesh.ny, mesh.nx))
    dx = (mesh.xFin - mesh.xStart)/mesh.nx * 1E3
    dy = (mesh.yFin - mesh.yStart)/mesh.ny * 1E3

    image = image*dx*dy*bandwidth*transmission/(1e-3)
    n = 1
    if norm is not False:
        if norm is True:
            image /= np.amax(image)
            n = np.amax(image)
        else:
            image /= norm
            n = norm

    x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    return image, x, y, mesh, n

def read_srw_phase_dat(file_name, unwrap=False):

    image, mesh = srwl_uti_read_intens_ascii(file_name)
    image = np.reshape(image, (mesh.ny, mesh.nx))

    if unwrap:
        image = unwrap_phase(image, wrap_around=False)

    x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    return image, x, y, mesh


def main():

    print('ideal case')
    file_name = './caustics/ID01_4xBe_CRL_ideal_9p0keV_d0p0mm_intensity_cstc_X.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e3, y * 1e6)
    image.legends = ['Horz. caustic', '(mm)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-50, 50, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=7.766563146, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_ideal_9p0keV_d0p0mm_intensity.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - intensity', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=4.8, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_ideal_9p0keV_d0p0mm_phase.dat'
    cstx, x, y, mesh = read_srw_phase_dat(file_name, unwrap=False)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - phase', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-np.pi, np.pi]
    image.ColorScheme = 2
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=True, Silent=False, m=4.8, n=4.8, isphase=True)

    print('aberrated case')
    file_name = './caustics/ID01_4xBe_CRL_ind_lens_mtrl_9p0keV_d0p0mm_intensity_cstc_X.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e3, y * 1e6)
    image.legends = ['Horz. caustic', '(mm)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-50, 50, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=7.766563146, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_ind_lens_mtrl_9p0keV_d0p0mm_intensity.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - intensity', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=4.8, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_ind_lens_mtrl_9p0keV_d0p0mm_phase.dat'
    cstx, x, y, mesh = read_srw_phase_dat(file_name, unwrap=False)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - phase', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-np.pi, np.pi]
    image.ColorScheme = 2
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=4.8, n=4.8, isphase=True)

    print('aberrated case - bis')
    file_name = './caustics/ID01_4xBe_CRL_stack_mtrl_9p0keV_d0p0mm_intensity_cstc_X.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e3, y * 1e6)
    image.legends = ['Horz. caustic', '(mm)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-50, 50, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=7.766563146, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_stack_mtrl_9p0keV_d0p0mm_intensity.dat'
    cstx, x, y, mesh, Imax = read_srw_intensity_dat(file_name, norm=True)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - intensity', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-0.05, 1.05]
    image.ColorScheme = 5
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=False, Silent=False, m=4.8, n=4.8)

    file_name = './caustics/ID01_4xBe_CRL_stack_mtrl_9p0keV_d0p0mm_phase.dat'
    cstx, x, y, mesh = read_srw_phase_dat(file_name, unwrap=False)
    image = b4pt.Image2Plot(cstx, x * 1e6, y * 1e6)
    image.legends = ['PSF - phase', '($\mu$m)', '($\mu$m)']
    image.LaTex = True
    image.AspectRatio = False
    image.plt_limits = [-np.pi, np.pi]
    image.ColorScheme = 2
    image.FontsSizeScale = 1.3
    image.ax_limits = [-5, 5, -5, 5]
    image.Scale = 0
    image.sort_class()
    b4pt.plot_2D_cuts(image, file_name.replace('.dat', '.png'), Enable=True, Silent=False, m=4.8, n=4.8, isphase=True)


if __name__ == '__main__':
    main()
