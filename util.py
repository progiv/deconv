import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d #fftconvolve
from scipy.ndimage import filters
from numpy.fft import fftn, ifftn
from itertools import chain
from skimage.draw import line_aa
from skimage.measure import compare_psnr
from math import sin, cos, pi, factorial

def edgetaper(I, psf):
    """
    J = EDGETAPER(I,PSF) blurs the edges of image I using the point-spread
    function, PSF. The output image J is the weighted sum of the original
    image I and its blurred version. The weighting array, determined by the
    autocorrelation function of PSF, makes J equal to I in its central
    region, and equal to the blurred version of I near the edges.
    
    The EDGETAPER function reduces the ringing effect in image deblurring
    methods that use the discrete Fourier transform, such as DECONWNR,
    DECONVREG, and DECONVLUCY.
    
    Note that the size of the PSF cannot exceed half of the image size in any
    dimension.
    """
    # [I, PSF, sizeI, classI, sizePSF, numNSdim] = parse_inputs(varargin{:});
    if np.all(psf >= 0):
        psf /= psf.sum()
    # PSF size cannot be larger than sizeI/2 because windowing is performed
    # with PSF autocorrelation function
    if np.any(np.array(I.shape) <= np.array(psf.shape) * 2):
        raise ValueError(psf)

    numDim = len(psf.shape)
    beta = [0] * numDim
    for n in range(numDim):
        PSFproj = np.sum(psf, axis=n)
        z = np.real(ifftn(np.abs(fftn(PSFproj, (I.shape[n] - 1,)))**2))
        z = z/np.max(z)
        beta[n] = np.append(z, z[0])
    
    if numDim == 1:
        alpha = 1 - beta[0]
    else: # n == 2:
        alpha = np.matmul((1-beta[0]).reshape((I.shape[0], 1)), (1 - beta[1]).reshape(1, I.shape[1]))
    # n > 2 - not implemented
    
    blurred = convolve2d(I, psf, 'same', 'wrap')#, 'symm')
    J = alpha * I + (1-alpha)*blurred

    return np.clip(J, np.min(I), np.max(I))

def show_results(original, noisy, result, titles=['Original Data', 'Blurred data', 'Restoration using\nRichardson-Lucy']):
    # Show results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
           a.axis('off')

    ax[0].imshow(original)
    ax[0].set_title(titles[0])

    ax[1].imshow(noisy)
    ax[1].set_title(titles[1])

    ax[2].imshow(result)#, vmin=noisy.min(), vmax=noisy.max())
    ax[2].set_title(titles[2])


    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return filters.gaussian_filter(inp, nsig)

def plot_corr(n, ydata, legend=['row correlation', 'column correlation'], title='Normalized correlation'):
    plt.plot(*list(chain(*[(range(n), y) for y in ydata])))
    plt.grid()
    plt.title(title)
    plt.xlabel('n iterations')
    plt.legend(legend)
    plt.show()

def motion_blur_psf(length, angle):
    cc, rr, val = line_aa(0, 0, int(length*cos(angle)), int(length*sin(angle)))
    psf = np.zeros((max(rr)+1, max(cc)+1))
    psf[rr, cc] = val
    return psf/np.sum(psf)

def motion_blur_psf_my(length=10, angle=pi/4, n_points=1000, **kwargs):
    """Incorrectly works with angles k*pi/2, where k is integer."""
    x_start, y_start = 0, 0
    if 'x' in kwargs and 'y' in kwargs:
        x_end, y_end = kwargs['x'], kwargs['y']
    else:
        x_end, y_end = length*cos(angle), length*sin(angle)
    if x_end < 0:
        x_start -= x_end
        x_end = 0
    if y_end < 0:
        y_start -= y_end
        y_end = 0
    psf = np.zeros((int(max(y_start, y_end))+2, int(max(x_start, x_end))+2))
    
    triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
    triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
    
    X = np.linspace(x_start, x_end, n_points)
    Y = np.linspace(y_start, y_end, n_points)
    x1 = np.floor(X).astype(np.int)
    x2 = x1+1
    y1 = np.floor(Y).astype(np.int)
    y2 = y1+1
    print(x_start, y_start, x_end,y_end)
    psf[y1, x1] += triangle_fun_prod(X - x1, Y - y1)
    psf[y2, x1] += triangle_fun_prod(X - x1, Y - y2)
    psf[y1, x2] += triangle_fun_prod(X - x2, Y - y1)
    psf[y2, x2] += triangle_fun_prod(X - x2, Y - y2)
    return psf/np.sum(psf)

def find_noise(image):
    N, M = image.shape
    F = fftn(image)
    part = .02
    area = F[int(N/2-N*part):int(N/2+N*part), int(M/2-M*part):int(M/2+M*part)]
    np.std
    np.var
    msv = np.sqrt(np.mean(np.abs(area) ** 2) / (N*M))
    I_res = ifftn(F-msv)
    NSR = msv**2 / np.var(I_res)
    return (msv, NSR)

def binomial(i, n):
    """Binomial coefficient"""
    return factorial(n) / float(
        factorial(i) * factorial(n - i))


def bernstein(t, i, n):
    """Bernstein polynom"""
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(t, points):
    """Calculate coordinate of a point in the bezier curve"""
    n = len(points) - 1
    x = y = 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    return int(x), int(y)


def bezier_curve_range(n, points):
    """Range of points in a curve bezier"""
    for i in range(n):
        t = i / float(n - 1)
        yield bezier(t, points)

def bezier_psf(points, n=100):
    print(points)
    curve = bezier_curve_range(n, points)
    r, c = zip(*[p for p in curve])
    psf = np.zeros((int(np.ceil(np.max(r)+1)), int(np.ceil(np.max(c)+1))))
    for rr, cc in zip(r,c):
        psf[rr, cc] += 1
#     psf[r,c] = 1
    psf /= psf.sum()
    return psf

def bezier_psf_aa(points, n=100):
    print(points)
    curve = bezier_curve_range(n, points)
    Y, X = zip(*[p for p in curve])
    psf = np.zeros((int(np.ceil(np.max(Y)+2)), int(np.ceil(np.max(X)+2))))

    triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
    triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
    triangle_fun_curve = lambda x, X: np.array([max(np.maximum(0, 1-np.abs(X-xx))) for xx in x])
    triangle_fun_curve2 = lambda x, y, X, Y: np.sqrt(triangle_fun_curve(x, X)**2  + triangle_fun_curve(y, Y)**2)

    x1 = np.floor(X).astype(np.int)
    x2 = x1+1
    y1 = np.floor(Y).astype(np.int)
    y2 = y1+1

    psf[y1, x1] += triangle_fun_curve2(X, Y, x1, y1)
    psf[y2, x1] += triangle_fun_curve2(X, Y, x1, y2)
    psf[y1, x2] += triangle_fun_curve2(X, Y, x2, y1)
    psf[y2, x2] += triangle_fun_curve2(X, Y, x2, y2)

    return psf/psf.sum()

def bezier_psf2(points, n=100):
    x = [0] + points[::2]
    y = [0] + points[1::2]
    xy = list(zip(x, y))
    return bezier_psf_aa(xy, n=n)

def compare_psnr_crop(im_true, im_test, crop_area=20, **kwargs):
    if crop_area is None:
        return compare_psnr(im_true, im_test, **kwargs)
    elif isinstance(crop_area, int):
        return compare_psnr(im_true[crop_area:-crop_area, crop_area:-crop_area],
                            im_test[crop_area:-crop_area, crop_area:-crop_area], **kwargs)
    else:
        return compare_psnr(im_true[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]],
                            im_test[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], **kwargs)
