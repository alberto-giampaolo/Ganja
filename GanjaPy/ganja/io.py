import numpy as np
import root_numpy as rnp
import pandas as pd

from skimage.transform import downscale_local_mean

import os
import json

# -------------------------------------------------------------------------------------------
def read_root(fnames,tree=None,inpshape=(120,120,1),
              imgen='jetImageGen',imreco='jetImageReco',
              rebin_as=None,
              post_process=None,
              **kwargs):

    if not 'branches' in kwargs:
        kwargs['branches'] = ['nPU','rho',
                              'ptGen','etaGen','phiGen',
                              'pt','eta','phi',
        ]
    keys = filter(lambda x: x not in [imreco,imgen], kwargs['branches'])

    if not imgen in kwargs['branches']:
        kwargs['branches'].append(imgen)
    if not imreco in kwargs['branches']:
        kwargs['branches'].append(imreco)

    arr = rnp.root2array(fnames,tree,**kwargs)
    if rebin_as is not None:
        factors = np.array(inpshape).astype(np.int) / np.array(rebin_as).astype(np.int)
        scale_back = factors.prod()
        factors = tuple(factors.tolist())
        def rebin(img):
            img = img.reshape(inpshape)
            ret=downscale_local_mean(img, factors)
            return ret
        gen = np.array(list(map(rebin, arr[imgen])))
        reco = np.array(list(map(rebin, arr[imreco])))
        gen *= scale_back
        reco *= scale_back
    else:
        gen = np.vstack(arr[imgen]).reshape( (-1,)+inpshape )
        reco = np.vstack(arr[imreco]).reshape( (-1,)+inpshape )

    cols = { x:arr[x] for x in keys}
    df = pd.DataFrame(data=cols)
    if post_process:
        gen,reco = post_process(df,gen,reco)
    return df,gen,reco

# -------------------------------------------------------------------------------------------
def write_out(dest,folder,fnum,record,df,gen,reco):

    dirname = "%s/%s" % (dest,folder)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    basename = "%s/%d" % (dirname,fnum)
    df.to_hdf(basename+'.hd5','info',format='t',mode='w',complib='bzip2',complevel=9)
    np.savez_compressed(basename+'_gen.npz',gen)
    np.savez_compressed(basename+'_reco.npz',reco)
    rec = open(basename+'.json','w+')
    rec.write(json.dumps( record  ))
    rec.close()
    
    
    

    