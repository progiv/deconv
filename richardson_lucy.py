import numpy as np
from scipy.signal import fftconvolve, convolve2d
from numpy.fft import fftn, ifftn
from scipy.stats.stats import pearsonr

def row_correlation(image):
    return pearsonr(image[1:,:].ravel(), image[:-1,:].ravel())[0]
    
def col_correlation(image):
    return pearsonr(image[:,1:].ravel(), image[:,:-1].ravel())[0]


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def corelucy(Y, H, dampar22, wI, readout, eps):
    """
    CORELUCY Accelerated Damped Lucy-Richarson Operator.
    Calculates function that when used with the scaled projected array
    produces the next iteration array that maximizes the likelihood that
    the entire suite satisfies the Poisson statistics.
    """
    reBlurred = np.real(ifftn(H *fftn(Y)))

    # 2. An Estimate for the next step
    reBlurred += readout;
    reBlurred[reBlurred <= 0] = eps;
    #reBlurred[reBlurred < eps] = eps;
    AnEstim = wI / reBlurred + eps;

    # 3. Damping if needed
    if dampar22 == 0: # No Damping
        ImRatio = AnEstim
    else: # Damping of the image relative to dampar22 = (N*sigma)^2
        gm = 10;
        g = (wI * np.log(AnEstim)+ reBlurred - wI) / dampar22;
        g = np.minimum(g,1);
        G = (g**(gm-1))*(gm-(gm-1)*g);
        ImRatio = 1 + G * (AnEstim - 1);
    return fftn(ImRatio)

def richardson_lucy_matlab(image, psf, iterations=50, dampar=0, weight=None, readout=0, 
                           eps=2.22e-16, clip=True):
    """ Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    """
    H = psf2otf(psf, image.shape)
    prev_image = image.copy()
    prev_prev_image = 0
    internal = np.zeros((image.size,2))
    internal[:,1] = 0

    if weight is None:
        weight = np.ones(image.shape)

    wI = np.maximum(weight * (readout + image), 0)
    scale = np.real(ifftn(np.conj(H)*fftn(weight))) + np.sqrt(eps)
    del weight

    dampar22 = np.square(dampar)/2

    # 3 L_R Iterations
    lambd = 2 * np.any(internal != 0)
    correlation_X = [col_correlation(image)]
    correlation_Y = [row_correlation(image)]
    for k in range(lambd+1, lambd+1 + iterations):
        # 3.a Make an image predictions for the next iteration
        if k > 2:
            lambd = (internal[:,0].T.dot(internal[:,1])) / (internal[:,1].T.dot(internal[:,1]) + eps)
            lambd = np.maximum(np.minimum(lambd, 1), 0) # stability enforcement saturation
        Y = np.maximum(prev_image + lambd*(prev_image - prev_prev_image), 0) # plus positivity constraint

        # 3.b Make core for the LR estimation
        CC = corelucy(Y, H, dampar22, wI, readout, eps)
        
        # 3.c Determine next iteration image and apply poitivity constraint
        prev_prev_image = prev_image
        prev_image = np.maximum(Y * np.real(ifftn(np.conj(H) * CC)) / scale, 0)#np.conj(psf)
        del CC
        if clip:
            prev_image[prev_image > 1] = 1
            #prev_image[prev_image < eps] = eps
        correlation_X.append(col_correlation(prev_image))
        correlation_Y.append(row_correlation(prev_image))
        internal[:,1] = internal[:,0]
        internal[:,0] = (prev_image-Y).ravel()
    del wI, scale, Y
    return {'image': prev_image, 
            'correlationX': np.array(correlation_X)/(image.size-1), 
            'correlationY': np.array(correlation_Y)/(image.size-1)}
