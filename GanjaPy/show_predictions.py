import GAN.models as models
import ganja.io as io
import ganja.utils as utils
import ganja.plotting as plotting
import ganja.preprocessing as preprocessing
import ganja.plotting as plotting

import json

import sys

import numpy as np
import pandas as pd

## from IPython import get_ipython

import matplotlib
matplotlib.style.use('seaborn-poster')


from  keras.models import load_model

## # target = 'log/rerun_unnorm_nt4_l2_sm_wtot_gtot_tau5m4_gtau5_df4_gan_lw1'
## # target = 'log/unnorm_l2_sm_gtot_gtau5_df4_gan_lw0p5'
## # target = 'log/unnorm_nt4_l2_sm_wtot_gtot_tau5m4_gtau5_df4_gan_lw1/'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau2m5_gtau4p5_df4_gan_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg6m8_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg6m8_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg6m8_lw1'

## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg1m7_lw1'
## # target = 'log/unnorm_nt4_l2_sm_wtot_gtot_tau5m4_gtau5_df4_gan_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg6m8_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg1m2_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg1m5_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg1m3_lw1'
## # target = 'log/rerun_unnorm__nt4_nob_l2_sm_wtot_gtot_tau1m5_gtau5_df4_gan_lrdecg1m7_lw1'
target = 'log/rerun_unnorm__nd100_nt4_nob_l2_sm_wtot_gtot_tau2m5_gtau5_df4_gan_lw1'
## # target = 'log/rerun_unnorm__nd50_nt4_nob_l2_sm_wtot_gtot_tau2m5_gtau5_df4_gan_lw1'


import os
try :
    os.makedirs('/scratch/snx3000/albertog/'+target)
except:
    print("perbacco")
## target = sys.argv[1]

with open(target+'/config.json') as fin:
    training_parameters = json.loads(fin.read())
    
globals().update(training_parameters)

training_parameters

import os

with open(BASE_FOLDER+'/'+VERSION+'/train_valid_test.json') as fin:
    split_map = json.loads(fin.read())


test_inputs = {os.path.join(BASE_FOLDER,VERSION) : split_map['test'][:1]}

moments = np.load(MOMENTS.format(VERSION=VERSION))
reco_moments = moments['reco_mean'],moments['reco_std'],0.02
gen_moments = moments['gen_mean'],moments['gen_std'],0.02

weights = np.load(WEIGHTS.format(VERSION=VERSION),encoding='latin1')
weights = weights[0],weights[2:],weights[1]


test_reader = io.Reader(test_inputs,weights,compressed=COMPRESSED,cond_names=COND_VARIABLES,noise_dim=NOISE_DIM,
                       )#aux_noise=STOCHASTIC_LAYER)#,gen_moments=gen_moments)


ret = test_reader.get()
X_test,y_test = ret[0:2]
w_test = ret.pop(-1)

if test_reader.aux_noise:
    ret[-1] = ret[-1][:,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE]
inputs_test = [X_test[:,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE]]+ret[2:]

X_unnorm = X_test[:,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE]
y_test = y_test[:,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE]

previous = None

## import subprocess
## proc = subprocess.Popen('ls %s/model*[0-9].hdf5 | sort -n -r -t - -k 2' % target, stdout=subprocess.PIPE)
## weights = proc.stdout.read().split("\n")
## weights = ! ls $target/model*[0-9].hdf5 | sort -n -r -t - -k 2

## print("\n".join(weights))

## latest = weights[0]
## epoch = latest.split("-")[1].replace(".hdf5","")
## print('\n\nlatest '+latest)

## epoch = "10"
## run_inputs = [ x for x in inputs_test ]
run_inputs = [ x[:100000] for x in inputs_test ]
## print(run_inputs)

def generate_jets(pred_val,pred_prob,rescale=True):
    pred = pred_val*(np.random.uniform(size=pred_val.shape) < pred_prob)
    if rescale:
        pred_scl = np.sum(pred_val*pred_prob,axis=(1,2),keepdims=True)/np.sum(pred,axis=(1,2),keepdims=True)
        pred *= pred_scl
    return pred
    

# for epoch in ["10","9","11","8","12"]:
for epoch in map(str, range(1,15)):
    print(epoch)
    
    latest = "%s/model-%s.hdf5" % (target,epoch)

    import keras.backend as K
    import ganja.losses
    
    ## reload(ganja.losses)
    from GAN.unet import StochasticThreshold
    print('loading model '+latest)
    model = load_model(latest,custom_objects=dict(StochasticThreshold=StochasticThreshold),compile=False)
    previous = latest
    
    print(getattr(model,"loss",None))
    
    raw_pred = model.predict(run_inputs)
    
    if type(raw_pred) == list:
        y_pred = raw_pred[1]
        y_pred0 = raw_pred[0]
    else:
        y_pred = raw_pred
        y_pred0 = None
    
    if y_pred.shape[-1] >1:
        y_pred_val,y_pred_prob = y_pred[:,:,:,0:1],y_pred[:,:,:,1:]
    else:
        y_pred_val,y_pred_prob = y_pred,None
    
    gen = X_unnorm[:y_pred_val.shape[0],WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE,:]
    reco = y_test[:y_pred_val.shape[0],WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE,:]
    if y_pred_prob is not None:
        pred  = generate_jets(y_pred_val,y_pred_prob,False)
    
    from matplotlib.colors import LogNorm 
    
    rings, ring_masks = plotting.make_masks([0.3, 0.2, 0.15, 0.1, 0.05, 0.],npix=IMG_SIZE/2,rad=0.3*IMG_SIZE/32.)
    
    reco_rings = []
    pred_rings = []

    for mask in ring_masks:
        reco_rings.append( reco[:,mask].sum(axis=(1,2)) )
        pred_rings.append( pred[:,mask].sum(axis=(1,2)) )
    
    print('reco_rings',len(reco_rings))
    
    if len(inputs_test) == 2:
        cond = inputs_test[1][:pred.shape[0]]
    else:
        cond = inputs_test[2][:pred.shape[0]]
        weights = w_test[:pred.shape[0]]
    
    
    pts = cond[:,0,0]
    etas = cond[:,0,1]
    phis = cond[:,0,2]
    npus = cond[:,0,3]
    
    nj = pred.shape[0]
    ## print(nj)
    steps = min(200,nj)
    
    ##  sub_reco = plotting.compute_substructure(y_test[:nj,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE],pts[:nj],etas[:nj],phis[:nj],steps=steps)
    ##  sub_pred = plotting.compute_substructure(pred[:nj,WIN_MIN:WIN_MIN+IMG_SIZE,WIN_MIN:WIN_MIN+IMG_SIZE],pts[:nj],etas[:nj],phis[:nj],steps=steps)
    ##  
    ##  tau32_reco = (sub_reco[:,6] / sub_reco[:,5]).reshape(-1,1)
    ##  tau32_pred = (sub_pred[:,6] / sub_pred[:,5]).reshape(-1,1)
    ##  
    ##  sub_reco = np.concatenate([sub_reco,tau32_reco],axis=-1)
    ##  sub_pred = np.concatenate([sub_pred,tau32_pred],axis=-1)
    ##  
    ##  sub_reco[:,1:3] = np.exp(-sub_reco[:,1:3])
    ##  sub_pred[:,1:3] = np.exp(-sub_pred[:,1:3])
    
    
    columns={"pt" : pts, "eta": etas, "phi": phis, "npu": npus }
    for iring in range(len(reco_rings)):
        columns.update( { "reco_ring%d" %iring : reco_rings[iring], "pred_ring%d"  %iring : pred_rings[iring] }  ) 
        print(iring)
        print(columns.keys())
        columns["reco_total"] = reco.sum(axis=(1,2,3))
        columns["pred_total"] = pred.sum(axis=(1,2,3))
    
    
    sub_columns = {}
    ##sub_columns = { 0 : "ptD", 1: "maja", 2 : "mina", 
    ##                3 : "tau21", 4 : "tau1", 5 : "tau2", 
    ##                6 : "tau3", 7 : "tau32" }
    
    for col,name in sub_columns.items():
        columns["reco_%s" % name] = sub_reco[:,col]
        columns["pred_%s" % name] = sub_pred[:,col]
    
    df = pd.DataFrame( columns )
    ## df.to_hdf("%s/valid_50k_%s.hd5" % (target,epoch), key='valid' )
    df.to_hdf("%s/valid_100k_%s.hd5" % ('/scratch/snx3000/albertog/'+target,epoch), key='valid' )
