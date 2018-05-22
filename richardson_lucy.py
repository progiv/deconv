import numpy as np
from scipy.signal import fftconvolve, convolve2d
from scipy.stats.stats import pearsonr

def row_correlation(image):
    return pearsonr(image[1:,:].ravel(), image[:-1,:].ravel())[0]
    
def col_correlation(image):
    return pearsonr(image[:,1:].ravel(), image[:,:-1].ravel())[0]


def corelucy(Y, psf, dampar22, wI, readout, eps=1e-9, convolvemethod=convolve2d):
    """
    CORELUCY Accelerated Damped Lucy-Richarson Operator.
    Calculates function that when used with the scaled projected array
    produces the next iteration array that maximizes the likelihood that
    the entire suite satisfies the Poisson statistics.
    """
    reBlurred = np.real(convolvemethod(Y, psf, 'same'))

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
    return ImRatio

def richardson_lucy_matlab(image, psf, iterations=50, dampar=0, weight=None, readout=0, 
                           eps=1e-16, clip=True, useFFT=False):
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
    psf_mirror = psf[::-1,::-1]
    
    assert(not np.all(psf == 0))

    if isinstance(image, list) and len(image) == 4:
        image, prev_image, prev_prev_image, internal = image
    else:
        prev_image = image.copy()#np.zeros(image.shape)*.5#image
        prev_prev_image = 0
        internal = np.zeros((image.size,2))
    internal[:,1] = 0

    if weight is None:
        weight = np.ones(image.shape)

    wI = np.maximum(weight * (readout + image), 0)
    if useFFT:
        convolvemethod = fftconvolve
    else:
        convolvemethod = convolve2d
    scale = convolve2d(weight, psf, 'same') + np.sqrt(eps)
    del weight

    dampar22 = np.square(dampar)/2

    # 3 L_R Iterations
    lambd = 2 * np.any(internal != 0)
    correlation_X = [col_correlation(image)]
    correlation_Y = [row_correlation(image)]
    for k in range(lambd, lambd + iterations):
        # 3.a Make an image predictions for the next iteration
        if k > 2:
            lambd = (internal[:,0].T.dot(internal[:,1])) / (internal[:,1].T.dot(internal[:,1]) + eps) # (scalar division)
            lambd = np.maximum(np.minimum(lambd, 1), 0) # stability enforcement saturation
        Y = np.maximum(prev_image + lambd*(prev_image - prev_prev_image), 0) # plus positivity constraint

        # 3.b Make core for the LR estimation
        cc = corelucy(Y, psf, dampar22, wI, readout, eps, convolvemethod)
        
        # 3.c Determine next iteration image and apply poitivity constraint
        prev_prev_image = prev_image
        prev_image = np.maximum(Y * np.real(convolvemethod(cc, psf_mirror, 'same')) / scale, 0)#np.conj(psf)
        if clip:
            prev_image[prev_image > 1] = 1
            prev_image[prev_image < eps] = eps
        correlation_X.append(col_correlation(prev_image))
        correlation_Y.append(row_correlation(prev_image))
        del cc
        internal[:,1] = internal[:,0]
        internal[:,0] = (prev_image-Y).ravel()
    del wI, scale, Y
    return {'image': prev_image, 
            'correlationX': np.array(correlation_X)/(image.size-1), 
            'correlationY': np.array(correlation_Y)/(image.size-1)}
