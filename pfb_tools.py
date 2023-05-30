import numpy as np

def get_win(n:int,ntap:int,hamming=True):
    """Gets the sinc window array with optional hamming taper.

    Parameters
    ----------
    n : int
        The length of a frame, usually a power of 2 e.g. 2048
    ntap : int
        The number of taps, does not have to be a power of 2.
    hamming : bool
        Whether or not to taper the sinc with a hamming function.

    Returns
    -------
    1d-array
        The window's weights
    """
    #x=np.linspace(-1,1,n*ntap)
    x=np.arange(n*ntap)/(n*ntap/2)-1
    #asdf=x*ntap/2
    #print(asdf[:5])
    mysinc=np.sinc(x*ntap/2)
    if hamming:
        #vec=0.5+0.5*np.cos(np.pi*x)
        vec=np.hamming(len(x))
    else:
        vec=1
    return mysinc*vec
    
def get_fft_mat(n:int,kmin:int=0,kmax:int=None):
    """Get the full or partial FFT matrix operator as a 2d-array.

    Parameters
    ----------
    n : int
        Length of input array
    kmin : int 
        Lowest frequency number to compute
    kmax : int
        Highest frequency number to compute

    Returns
    -------
    2d-array
        A numpy 2d array of shape (kmax + 1 - kmin, n)
        If we matrix-multiply this array with a 2d-vec it will output
        the modes (kmin, kmin+1, ..., kmax) of the FFT. 
    """
    if kmax is None:
        kmax=n-1 
    xvec=np.arange(n) # 1d array, positive integers from 0 to n-1
    kvec=np.arange(kmin,kmax+1) # 1d integer array, size depends on 
                                # which and how many frequencies we 
                                # would like to output
    kx=np.outer(kvec,xvec) # kx are the arguments of entries in FFT mat
    return np.exp(-2*np.pi*1J*kx/n) # (partial) FFT matrix



def get_pfb_mat(nchan:int, ntap:int, *args,**kwargs):
    """Get the matrix that represents the PFB operator on ntap frames.

    If we have 4 taps and our frames are of length 2048 = 2*(1025 - 1)
    this will return a matrix that 

    Parameters
    ----------
    nchan : int
        Number of frequency channels.
    ntap : int
        Number of taps, i.e. the factor by which to widen the input.
    kmin : int (optional)
        Passed to `get_fft_mat`
    kmax : int (optional)
        Passed to `get_fft_mat`

    Returns
    -------
    2d-array
        A matrix of shape (number of frequency modes , n * ntap), where
        n is the number of frames, n=2*(nchan - 1) and m, the number of
        frequency modes is (kmax + 1 - kmin) which is n by default.
    """
    n=2*(nchan-1) # length of each frame is equal to 2 * (nchan - 1)
    win=get_win(n,ntap)
    #fft_mat=get_fft_mat(n,kmax=nchan-1)
    fft_mat=get_fft_mat(n,*args,**kwargs) # Get the FFT matrix.
    # NB: passing *args and **kwargs is a fancy pythonic way to pass 
    # arbitrary numbers of (named or un-named) optional parameters, in 
    # this case we optionally pass the kmin and kmax parameters if needed
    mats=[None]*ntap # a list of length ntap where every item is None
    m=fft_mat.shape[0] # the shape of the output
    for i in range(ntap):
        # copy and stack window segments, shape (m, n)
        tmp=np.repeat([win[i*n:(i+1)*n]],m,axis=0) 
        mats[i]=fft_mat*tmp
    return np.hstack(mats) # Horizontally stack matrices


def pfb_chunk(seg:np.ndarray,win:np.ndarray,ntap:int=4):
    """Performs the PFB operation on a segment of data.
    
    In the matrix formalism of the PFB, the action of the PFB on ntap 
    frames can be represented by the matrices FSW, where:
    
    W is the diagonal weights matrix, 
    S is a horizontal stack of identity matrices, and
    F is the r2c DFT. 
    
    Parameters
    ----------
    seg : 1d-array
        A segment or 'frame' of raw data to be channelized (PFB'd). 
    win : 1d-array
        The window and taper function weights (usually sinc-hamming)
    ntap : int
        Number of taps used.
    """
    assert seg.shape==win.shape, "Len window must equal len of frame"
    seg=seg*win # Apply window and taper function, (W)
    seg=np.reshape(seg,[ntap,len(seg)//ntap]) # Split, stack, [sum] (S)
    return np.fft.rfft(np.sum(seg,axis=0)) # sum, then DFT (F)
    
def pfb_data(dat:np.ndarray,nchan:int,ntap:int=4):
    """Performs PFB on a long piece of data. 
    
    Parameters
    ----------
    dat : 1d-array
        Data to be channelized.
    nchan : int
        Number of channels in the output of the PFB.
    ntap : int
        Number of taps to use, four by default. 
        
    Returns
    -------
    2d-array
        The channelized output, a numpy 2d array (type numpy.ndarray)
    """
    n=2*(nchan-1)       # length of a frame
    win=get_win(n,ntap) # get the window weights (i.e. sinc-hamming)
    nchunk=int((len(dat)/n-(ntap-1))) # number of times we apply FSW
    out=np.empty([nchunk,nchan],dtype='complex') 
    # compute PFB
    for i in range(nchunk):
        out[i,:]=pfb_chunk(dat[i*n:(i*n+ntap*n)],win,ntap)
    return out # return channelized data


def osamp_matrix(mat:np.ndarray, nrep:int, ntap:int=4):
    """Over sample the matrix mat. 
    
    Construct mat that applies it on many frames of data.
    ------------------------------------
    |    mat    |                      |
    ------------------------------------
    |  |    mat    |                   |
    ------------------------------------
    |    |    mat    |                 |
    ------------------------------------
    |      |    mat    |               |
    ------------------------------------
    |        |    mat    |             |
    ------------------------------------
    |          |    mat    |           |
    ------------------------------------
    |            |    mat    |         |
    ------------------------------------
    |              |    mat    |       |
    ------------------------------------
    |                |    mat    |     |
    ------------------------------------
    |                  |    mat    |   |
    ------------------------------------
    |                    |    mat    | |
    ------------------------------------
    |                      |    mat    |
    ------------------------------------
 
    
    Parameters
    ----------
    mat : 2d-array
        A 2d matrix (think PFB matrix). 
        Shape=(num modes to compute , num of taps * length of a frame)
    nrep : int
        Number of times to repeat the operation, i.e. the number of 
        times to apply PFB on sliding window of frames.
    ntap : int
        Number of taps. 
        
    Returns
    -------
    2d-array
        A matrix representing the PFB (or mat) acting on many frames of
        data. 
    """
    dx=mat.shape[1]//ntap # Length of a frame
    m=mat.shape[0] # Number of modes or channels
    n=mat.shape[1] # The length of a vector to be input into matrix
    mat_out=np.zeros([m*nrep,n+dx*(nrep-1)],dtype=mat.dtype)
    for i in range(nrep):
        mat_out[i*m:(i+1)*m,i*dx:(i*dx+n)]=mat
    return mat_out

def quantize_matrix(mat:np.ndarray,nlevel:int=15):
    """Quantize the matrix to n levels. 
    
    Assumes values of mat centered around zero. For both real and 
    complex input, we find the largest real or imag absolute value, 
    name it `a`, divide the matrix by `a` to scale it between -1 and 1
    in both real and imaginary componants, then multilply fac and round
    each entry to nearest (gaussian) integer.
    
    Parameters
    ----------
    mat : 2d-array
        The matrix that we would like to quantize. 
    nlevel : int
        Number of quantization levels. If we 8-bit quantize this is 15.
        
    Returns
    -------
    2d-array
        The quantized matrix. All its entries are integers, so it is 
        also scaled by some factor.
    """
    fac=nlevel/2.001
    # If any entry in the matrix is complex, quantize both real an imag
    if np.any(np.iscomplex(mat)):
        a=np.max(np.abs(np.real(mat)))
        b=np.max(np.abs(np.imag(mat)))
        if b>a:
            a=b 
        # quantize matrix
        out=np.round(np.real(mat)/a*fac)+np.round(np.imag(mat)/a*fac)*1J
    # If the data is real, just quantize that data to nlevel levels
    else:
        a=np.max(np.abs(mat))
        out=int(np.round(mat/a*fac)) # quantize
    return out # a matrix of integers, (data-type not necessarily ints)
