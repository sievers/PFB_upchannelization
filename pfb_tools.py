import numpy as np

def get_win(n,ntap,hamming=True):
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
    
def get_fft_mat(n,kmin=0,kmax=None):
    if kmax is None:
        kmax=n-1
    xvec=np.arange(n)
    kvec=np.arange(kmin,kmax+1)
    kx=np.outer(kvec,xvec)
    return np.exp(-2*np.pi*1J*kx/n)



def get_pfb_mat(nchan,ntap,*args,**kwargs):
    n=2*(nchan-1)
    win=get_win(n,ntap)
    #fft_mat=get_fft_mat(n,kmax=nchan-1)
    fft_mat=get_fft_mat(n,*args,**kwargs)
    mats=[None]*ntap
    m=fft_mat.shape[0]
    for i in range(ntap):
        tmp=np.repeat([win[i*n:(i+1)*n]],m,axis=0)
        mats[i]=fft_mat*tmp
    return np.hstack(mats)


def pfb_chunk(seg,win,ntap=4):
    seg=seg*win
    seg=np.reshape(seg,[ntap,len(seg)//ntap])
    #print(seg.shape)
    return np.fft.rfft(np.sum(seg,axis=0))
    
def pfb_data(dat,nchan,ntap=4):
    n=2*(nchan-1)
    win=get_win(n,ntap)
    nchunk=int((len(dat)/n-(ntap-1)))
    out=np.empty([nchunk,nchan],dtype='complex')
    for i in range(nchunk):
        out[i,:]=pfb_chunk(dat[i*n:(i*n+ntap*n)],win,ntap)
    return out


def osamp_matrix(mat,nrep,ntap=4):
    dx=mat.shape[1]//ntap
    m=mat.shape[0]
    n=mat.shape[1]
    mat_out=np.zeros([m*nrep,n+dx*(nrep-1)],dtype=mat.dtype)
    for i in range(nrep):
        mat_out[i*m:(i+1)*m,i*dx:(i*dx+n)]=mat
    return mat_out

def quantize_matrix(mat,nlevel=15):
    fac=nlevel/2.001
    if np.any(np.iscomplex(mat)):
        a=np.max(np.abs(np.real(mat)))
        b=np.max(np.abs(np.imag(mat)))
        if b>a:
            a=b
        out=np.round(np.real(mat)/a*fac)+np.round(np.imag(mat)/a*fac)*1J
    else:
        a=np.max(np.abs(mat))
        out=int(np.round(mat/a*fac))
    return out
