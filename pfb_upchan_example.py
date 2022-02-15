#example of upchannelization from a single coarse PFB channel
#upchannelization coefficients are determined by minimizing the
#least-squares residual to the fine PFB
#after generating the filter matrix, we generate a chirp and
#apply the coarse PFB to it, then upchannelize and show
#the fine PFB
#NB - the filter matrix will work for multiple input coarse
#channels, but the bit that applies the matrix needs some tweaking
#in the indexing if you want to use multiple coarse channels.

import numpy as np
import pfb_tools
from matplotlib import pyplot as plt
plt.ion()


#set up basic parameters for the upchannelization we want to do.
nchan=1025
nn=2*(nchan-1)

ntap_coarse=4 #how many taps to use in the coarse PFB
ntap_fine=2  #how many taps to use in the fine PFB
osamp=16 #this is the factor by which we wish to upchannelize

#pick a random coarse channel (range).  It should not 
#matter which channel one picks (away from the first/last)
#but this is to be verified.  Then produce the matrix that
#converts time samples to the coarse PFB for the desired channel(s)
kmin=344
kmax=344
mat=pfb_tools.get_pfb_mat(nchan,ntap_coarse,kmin=kmin,kmax=kmax)

#find the matrix that converts time samples to the fine PFB
kkmin=kmin*osamp-osamp//2
kkmax=kmax*osamp+osamp//2-1
mat_fine=pfb_tools.get_pfb_mat(1+(nchan-1)*osamp,ntap_fine,kmin=kkmin,kmax=kkmax)

#figure out how many coarse blocks you need to match the fine, since
#the coarse needs to be repeated
n_repeat=int((mat_fine.shape[1]-mat.shape[1])/(2*(nchan-1)))+1
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

#we'll generate a chirp that goes from coarse channels k0 to k1
k0=60
k1=70

nsamp=2**24 #number of samples in the chirp

dt=1+np.linspace(0,(k1/k0)-1,nsamp) #let's make a frequency sweep
tvec=np.cumsum(dt)
dat=np.sin(2*np.pi*k0/nn*tvec)

#the PFB of our chirp
pfb_coarse=pfb_tools.pfb_data(dat,nchan,ntap_coarse)

#roughly how many output blocks of the fine PFB we're going to have
nn_fine=nn*osamp
nchunk_fine=pfb_coarse.shape[0]//osamp-ntap_fine #this is extra conservative
#i=0
#flub=pfb_coarse[(osamp*i):(osamp*(i+ntap_fine)),:]

#how many coarse blocks go into one fine block
n_mult=osamp*ntap_fine-ntap_coarse+1
n_mult=n_mult*(kmax-kmin+1)
print(filt_mat.shape[0],n_mult)

#this comprehension is looping over blocks in the coarse PFB and 
#multiplying by the upchannelization matrix.  
pfb_fine=[filt_mat.T@pfb_coarse[(osamp*i):(osamp*i+n_mult),:] for i in range(nchunk_fine)]

#each output block is now nfine by nchan, so we want to ravel them to get into single time slices
#right now the fast index is coarse frequency, so take a transpose before ravelling
pfb_fine=[np.ravel(block.copy().T) for block in pfb_fine]

#finally, turn into an array
pfb_fine=np.asarray(pfb_fine)

#show our coarse PFB
plt.figure(1)
plt.clf();
plt.imshow(np.abs(pfb_coarse[:,k0:k1]));
plt.axis('auto');
plt.show()

#show our upchannelized data
plt.figure(2)
plt.clf()
plt.imshow(np.abs(pfb_fine[:,(k0*osamp):(k1*osamp)]));
plt.axis('auto');
plt.show()
