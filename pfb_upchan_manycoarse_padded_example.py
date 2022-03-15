
import numpy as np
import pfb_tools
from matplotlib import pyplot as plt
plt.ion()


#set up basic parameters for the upchannelization we want to do.
nchan=513
nn=2*(nchan-1)

ntap_coarse=4 #how many taps to use in the coarse PFB
ntap_fine=4  #how many taps to use in the fine PFB
osamp=8 #this is the factor by which we wish to upchannelize

#pick a random coarse channel (range).  It should not 
#matter which channel one picks (away from the first/last)
#but this is to be verified.  Then produce the matrix that
#converts time samples to the coarse PFB for the desired channel(s)
kmin=344
kmax=345
mat=pfb_tools.get_pfb_mat(nchan,ntap_coarse,kmin=kmin,kmax=kmax)
ncoarse=kmax-kmin+1


#find the matrix that converts time samples to the fine PFB
kkmin=kmin*osamp-osamp//2
kkmax=kmax*osamp+osamp//2-1
mat_fine=pfb_tools.get_pfb_mat(1+(nchan-1)*osamp,ntap_fine,kmin=kkmin,kmax=kkmax)

kmin2=kmin+ncoarse
kmax2=kmax+ncoarse
kkmin2=kmin2*osamp-osamp//2
kkmax2=kmax*osamp+osamp//2-1
mat_fine2=pfb_tools.get_pfb_mat(1+(nchan-1)*osamp,ntap_fine,kmin=kkmin2,kmax=kkmax2)


#figure out how many coarse blocks you need to match the fine, since
#the coarse needs to be repeated
aa=osamp*nn*ntap_fine
bb=nn*ntap_coarse
n_repeat=(aa-bb)//nn+1
mat_osamp=pfb_tools.osamp_matrix(mat,n_repeat,ntap_coarse)

#make the filter matrix: (A^T A + qI)^{-1}(A^T B) where A is the matrix
#that turns timestreams into the coarse PFB, B is the matrix that turns
#timestreams into the fine (upchannelized) PFB, and q is the quantization
#noise in the coarse PFB
lhs=np.conj(mat_osamp)@(mat_osamp.T)
rhs=np.conj(mat_osamp)@(mat_fine.T)
nbit=4 #number of bits we expect in the coarsely channelized PFB
amp=np.mean(np.diag(lhs))/(4**nbit)

#this is the matrix that will upchannelize the coarse PFB
#actually, it's the transpose of the matrix, given how the 
#other matrices were generated
filt_mat=np.linalg.inv(lhs+amp*np.eye(lhs.shape[0]))@rhs
#these coefficients go directly from the timestream samples to the
#upchannelized PFB output
ts_coeffs=(filt_mat.T)@mat_osamp

#we'll generate a chirp that goes from coarse channels k0 to k1
#k0=60
#k1=70
k0=343
k1=350

nsamp=2**22 #number of samples in the chirp

dt=1+np.linspace(0,(k1/k0)-1,nsamp) #let's make a frequency sweep
tvec=np.cumsum(dt)
dat=np.sin(2*np.pi*k0/nn*tvec)

#the PFB of our chirp
pfb_coarse=pfb_tools.pfb_data(dat,nchan,ntap_coarse)
pfb_fine_ref=pfb_tools.pfb_data(dat,nchan*osamp,ntap_fine)

nii=pfb_coarse.shape[0]//osamp-n_repeat-1
tmp2=[None]*nii
nuse=ncoarse-1
for iblock in range(nii):
    istart_coarse=iblock*osamp
    coarse_stripe=pfb_coarse[istart_coarse:istart_coarse+n_repeat,:]
    tmp=[None]*((nchan-ncoarse)//nuse)
    for k in range(0,(nchan-ncoarse)//nuse):
        kk=k*nuse
        tmp[k]=filt_mat.T@np.ravel(coarse_stripe[:,kk:kk+ncoarse])
        tmp[k]=tmp[k][(osamp//2+0):(-osamp//2+0)]
    tmp2[iblock]=np.ravel(np.asarray(tmp))

asdf=np.asarray(tmp2)
upchan_output=np.asarray(tmp2) #this is the upchannelized output
fwee=np.max(np.abs(upchan_output),axis=1)

#let's look at what we got.  In a perfect world, we would have a single
#diagonal stripe.
plt.figure(1)
plt.imshow(np.abs(upchan_output))
plt.axis('auto')
plt.xlim((k0-1)*osamp,(k1+1)*osamp) #this is where the signal ought to be

plt.figure(2)
plt.clf()
#this is the ideal output - note that there is still
#going to be scalloping, so we should not feel bad if our
#upchannelized version is scalloped
plt.plot(np.max(np.abs(pfb_fine_ref),axis=1))
#and here is the max signal we got.
plt.plot(fwee)
plt.ylim(0,plt.ylim()[1])
plt.show()
#print(np.std(fwee)/np.mean(fwee))
