# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 12:31:24 2020

@author: Adan
"""
#gen_loader
import numpy as np

def load_gen(struct, nu):
    if struct == 'cyl':
        filename = "gen1/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matxr.npy")[np.newaxis,5:-5,5:-5,:]#
        maty = np.load(filename+struct+nu+"matyr.npy")[np.newaxis,5:-5,5:-5,:]
        matz = np.load(filename+struct+nu+"matzr.npy")[np.newaxis,5:-5,5:-5,:]
        H1p = np.load(filename+struct+nu+"H1pr.npy")[np.newaxis,5:-5,5:-5,:]
        perm = np.load(filename+struct+nu+"permyr.npy")[np.newaxis,np.newaxis,5:-5,5:-5]
        cond = np.load(filename+struct+nu+"condyr.npy")[np.newaxis,np.newaxis,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
    if struct == 'cyl90':
        struct = 'cyl'
        filename = "gen1/"+'cyl'+nu+"/"
        matx = np.load(filename+struct+nu+"matxr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]#
        maty = np.load(filename+struct+nu+"matyr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        matz = np.load(filename+struct+nu+"matzr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        H1p = np.load(filename+struct+nu+"H1pr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        perm = np.load(filename+struct+nu+"permyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        cond = np.load(filename+struct+nu+"condyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 1, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 1, axes= (-2,-1))
        cond = np.rot90(cond, k = 1, axes= (-2,-1))
        perm = np.rot90(perm, k = 1, axes= (-2,-1))
        matx = np.rot90(matx, k = 1, axes= (-3,-2))
        maty = np.rot90(maty, k = 1, axes= (-3,-2))
        matz = np.rot90(matz, k = 1, axes= (-3,-2))
    if struct == 'cyl180':
        struct = 'cyl'
        filename = "gen1/"+'cyl'+nu+"/"
        matx = np.load(filename+struct+nu+"matxr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]#
        maty = np.load(filename+struct+nu+"matyr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        matz = np.load(filename+struct+nu+"matzr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        H1p = np.load(filename+struct+nu+"H1pr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        perm = np.load(filename+struct+nu+"permyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        cond = np.load(filename+struct+nu+"condyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 2, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 2, axes= (-2,-1))
        cond = np.rot90(cond, k = 2, axes= (-2,-1))
        perm = np.rot90(perm, k = 2, axes= (-2,-1))
        matx = np.rot90(matx, k = 2, axes= (-3,-2))
        maty = np.rot90(maty, k = 2, axes= (-3,-2))
        matz = np.rot90(matz, k = 2, axes= (-3,-2))
    if struct == 'cyl270':
        struct = 'cyl'
        filename = "gen1/"+'cyl'+nu+"/"
        matx = np.load(filename+struct+nu+"matxr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]#
        maty = np.load(filename+struct+nu+"matyr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        matz = np.load(filename+struct+nu+"matzr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        H1p = np.load(filename+struct+nu+"H1pr.npy")[np.newaxis]#,7:-8,7:-8,:2]#,5:-5,5:-5,:]
        perm = np.load(filename+struct+nu+"permyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        cond = np.load(filename+struct+nu+"condyr.npy")[np.newaxis,np.newaxis]#,7:-8,7:-8]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 3, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 3, axes= (-2,-1))
        cond = np.rot90(cond, k = 3, axes= (-2,-1))
        perm = np.rot90(perm, k = 3, axes= (-2,-1))
        matx = np.rot90(matx, k = 3, axes= (-3,-2))
        maty = np.rot90(maty, k = 3, axes= (-3,-2))
        matz = np.rot90(matz, k = 3, axes= (-3,-2))
    if struct == 'shape':
        filename = "gen2/"+struct+"s"+nu+"/"#[:,:,8:-9,8:-9]
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,8:-9,8:-9,0]#11:-11,11:-11,:2]#3:-4,3:-4,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,8:-9,8:-9,0]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,8:-9,8:-9,0]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,8:-9,8:-9,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,8:-9,8:-9]#11:-11,11:-11]#3:-4,3:-4]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,8:-9,8:-9]#11:-11,11:-11]#3:-4,3:-4]
        phH1p = np.zeros((1,1,H1p.shape[1],H1p.shape[2]))#H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,1,H1p.shape[1],H1p.shape[2]))
        for mm in range(1):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
    if struct == 'shape90':
        struct = 'shape'
        filename = "gen2/"+struct+"s"+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 1, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 1, axes= (-2,-1))
        cond = np.rot90(cond, k = 1, axes= (-2,-1))
        perm = np.rot90(perm, k = 1, axes= (-2,-1))
        matx = np.rot90(matx, k = 1, axes= (-3,-2))
        maty = np.rot90(maty, k = 1, axes= (-3,-2))
        matz = np.rot90(matz, k = 1, axes= (-3,-2))
    if struct == 'shape180':
        struct = 'shape'
        filename = "gen2/"+struct+"s"+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 2, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 2, axes= (-2,-1))
        cond = np.rot90(cond, k = 2, axes= (-2,-1))
        perm = np.rot90(perm, k = 2, axes= (-2,-1))
        matx = np.rot90(matx, k = 2, axes= (-3,-2))
        maty = np.rot90(maty, k = 2, axes= (-3,-2))
        matz = np.rot90(matz, k = 2, axes= (-3,-2))
    if struct == 'shape270':
        struct = 'shape'
        filename = "gen2/"+struct+"s"+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,3:-4,3:-4,:]#11:-11,11:-11,:2]#3:-4,3:-4,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,3:-4,3:-4]#11:-11,11:-11]#3:-4,3:-4]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 3, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 3, axes= (-2,-1))
        cond = np.rot90(cond, k = 3, axes= (-2,-1))
        perm = np.rot90(perm, k = 3, axes= (-2,-1))
        matx = np.rot90(matx, k = 3, axes= (-3,-2))
        maty = np.rot90(maty, k = 3, axes= (-3,-2))
        matz = np.rot90(matz, k = 3, axes= (-3,-2))
    if struct == 'bound':
        filename = "gen3/"+struct+nu+"/"#7:-8,7:-8
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,7:-8,7:-8,0]#10:-10,10:-10,:2]#2:-3,2:-3,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,7:-8,7:-8,0]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,7:-8,7:-8,0]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,7:-8,7:-8,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,7:-8,7:-8]#10:-10,10:-10]#,2:-3,2:-3]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,7:-8,7:-8]#10:-10,10:-10]#,2:-3,2:-3]
        phH1p = np.zeros((1,1,H1p.shape[1],H1p.shape[2]))#H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,1,H1p.shape[1],H1p.shape[2]))
        for mm in range(1):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
    if struct == 'bound90':
        struct = 'bound'
        filename = "gen3/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#2:-3,2:-3,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 1, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 1, axes= (-2,-1))
        cond = np.rot90(cond, k = 1, axes= (-2,-1))
        perm = np.rot90(perm, k = 1, axes= (-2,-1))
        matx = np.rot90(matx, k = 1, axes= (-3,-2))
        maty = np.rot90(maty, k = 1, axes= (-3,-2))
        matz = np.rot90(matz, k = 1, axes= (-3,-2))
    if struct == 'bound180':
        struct = 'bound'
        filename = "gen3/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#2:-3,2:-3,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 2, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 2, axes= (-2,-1))
        cond = np.rot90(cond, k = 2, axes= (-2,-1))
        perm = np.rot90(perm, k = 2, axes= (-2,-1))
        matx = np.rot90(matx, k = 2, axes= (-3,-2))
        maty = np.rot90(maty, k = 2, axes= (-3,-2))
        matz = np.rot90(matz, k = 2, axes= (-3,-2))
    if struct == 'bound270':
        struct = 'bound'
        filename = "gen3/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#2:-3,2:-3,:]#
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,2:-3,2:-3,:]#10:-10,10:-10,:2]#,2:-3,2:-3,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,np.newaxis,2:-3,2:-3]#10:-10,10:-10]#,2:-3,2:-3]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        absH1p = np.rot90(absH1p, k = 3, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 3, axes= (-2,-1))
        cond = np.rot90(cond, k = 3, axes= (-2,-1))
        perm = np.rot90(perm, k = 3, axes= (-2,-1))
        matx = np.rot90(matx, k = 3, axes= (-3,-2))
        maty = np.rot90(maty, k = 3, axes= (-3,-2))
        matz = np.rot90(matz, k = 3, axes= (-3,-2))
    if struct == 'dukx':
        filename = "gen4/"+struct+nu+"/"
        indx = 30
        indy = 21
        matx2 = np.load(filename+struct+nu+"matx.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        maty2 = np.load(filename+struct+nu+"maty.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        matz2 = np.load(filename+struct+nu+"matz.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        perm1 = np.load(filename+struct+nu+"permy.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        cond1 = np.load(filename+struct+nu+"condy.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        phH1p2 = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p2 = phH1p2.copy()
        cond2 = phH1p2.copy()
        perm2 = cond2.copy()
        for mm in range(H1p.shape[3]):
            phH1p2[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p2[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            cond2[0,mm,:,:] = cond1[0,:,:,mm]
            perm2[0,mm,:,:] = perm1[0,:,:,mm]
        phH1p = np.zeros((H1p.shape[3]-12,1,H1p.shape[1],H1p.shape[2]))
        absH1p = phH1p.copy()
        cond = phH1p.copy()
        perm = phH1p.copy()
        matx = np.zeros((H1p.shape[3]-12,H1p.shape[1],H1p.shape[2]))
        maty = np.zeros((H1p.shape[3]-12,H1p.shape[1],H1p.shape[2]))
        matz = np.zeros((H1p.shape[3]-12,H1p.shape[1],H1p.shape[2]))
        z = int(nu)-2
        if z >= 12: 
            z -= 10 
        j = 0
        for i in range(z,z+2):
            phH1p[j] = phH1p2[0,i]
            absH1p[j] = absH1p2[0,i]
            cond[j] = cond2[0,i+1]
            perm[j] = perm2[0,i+1]
            matx[j] = matx2[0,:,:,i]
            maty[j] = maty2[0,:,:,i]
            matz[j] = matz2[0,:,:,i]
            j +=1
        cond = cond[:,0,:,:][:,np.newaxis,:,:]
        perm = perm[:,0,:,:][:,np.newaxis,:,:]
    if struct == '1mmdukx':
        filename = "gen4/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,:,:,:]
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,:,:,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,:,:,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,:,:,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,:,:,:]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,:,:,:]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(H1p.shape[3]):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
    if struct == 'ella':
        filename = "gen5/"+struct+nu+"/"
        indx = 28
        indy = 14
        matx2 = np.load(filename+struct+nu+"matx.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        maty2 = np.load(filename+struct+nu+"maty.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        matz2 = np.load(filename+struct+nu+"matz.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        perm1 = np.load(filename+struct+nu+"permy.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        cond1 = np.load(filename+struct+nu+"condy.npy")[np.newaxis,indx:indx+47,indy:indy+47,:]
        phH1p2 = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p2 = phH1p2.copy()
        cond2 = np.zeros((1,cond1.shape[3],cond1.shape[1],cond1.shape[2]))
        perm2 = cond2.copy()
        for mm in range(H1p.shape[3]):
            phH1p2[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p2[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            cond2[0,mm,:,:] = cond1[0,:,:,mm]
            perm2[0,mm,:,:] = perm1[0,:,:,mm]
        phH1p = np.zeros((H1p.shape[3]-17,1,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        absH1p = phH1p.copy()
        cond = phH1p.copy()
        perm = phH1p.copy()
        matx = np.zeros((H1p.shape[3]-17,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        maty = np.zeros((H1p.shape[3]-17,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        matz = np.zeros((H1p.shape[3]-17,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        z = int(nu)-2
        j = 0
        for i in range(z,z+2):####SIZEEEEEE-1
            phH1p[j] = phH1p2[0,i+1] 
            absH1p[j] = absH1p2[0,i+1] 
            cond[j] = cond2[0,i+1] 
            perm[j] = perm2[0,i+1] 
            matx[j] = matx2[0,:,:,i+1] 
            maty[j] = maty2[0,:,:,i+1] 
            matz[j] = matz2[0,:,:,i+1]
            j += 1
        cond = cond[:,0,:,:][:,np.newaxis,:,:]
        perm = perm[:,0,:,:][:,np.newaxis,:,:]
    if struct == '1mmella':
        filename = "gen5/"+"ella"+nu+"/"
        matx2 = np.load(filename+struct+nu+"matx.npy")[np.newaxis,72:-72,40:-37,:]
        maty2 = np.load(filename+struct+nu+"maty.npy")[np.newaxis,72:-72,40:-37,:]
        matz2 = np.load(filename+struct+nu+"matz.npy")[np.newaxis,72:-72,40:-37,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,72:-72,40:-37,:]
        H1m = np.load(filename+struct+nu+"H1m.npy")[np.newaxis,72:-72,40:-37,:]
        perm1 = np.load(filename+struct+nu+"permy.npy")[np.newaxis,72:-72,40:-37,:]
        cond1 = np.load(filename+struct+nu+"condy.npy")[np.newaxis,72:-72,40:-37,:]
        phH1p2 = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p2 = phH1p2.copy()
        absH1m2 = phH1p2.copy()
        phH1m2 = phH1p2.copy()
        cond2 = np.zeros((1,cond1.shape[3],cond1.shape[1],cond1.shape[2]))
        perm2 = cond2.copy()
        for mm in range(H1p.shape[3]):
            phH1p2[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p2[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            phH1m2[0,mm,:,:] = np.unwrap(np.angle(H1m[0,:,:,mm],deg=False))
            absH1m2[0,mm,:,:] = np.abs(H1m[0,:,:,mm])
            cond2[0,mm,:,:] = cond1[0,:,:,mm]
            perm2[0,mm,:,:] = perm1[0,:,:,mm]
        phH1p = np.zeros((H1p.shape[3]-38,3,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        absH1p = phH1p.copy()
        absH1m = phH1p.copy()
        phH1m = phH1p.copy()
        cond = phH1p.copy()
        perm = phH1p.copy()
        matx = np.zeros((H1p.shape[3]-38,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        maty = np.zeros((H1p.shape[3]-38,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        matz = np.zeros((H1p.shape[3]-38,H1p.shape[1],H1p.shape[2]))####SIZEEEEEE
        z = int(nu)-2
        j = 0
        for i in range(z,z+1):####SIZEEEEEE-1
            phH1p[j] = phH1p2[0,i] 
            absH1p[j] = absH1p2[0,i] 
            phH1m[j] = phH1m2[0,i] 
            absH1m[j] = absH1m2[0,i] 
            cond[j] = cond2[0,i+1] 
            perm[j] = perm2[0,i+1] 
            matx[j] = matx2[0,:,:,i] 
            maty[j] = maty2[0,:,:,i] 
            matz[j] = matz2[0,:,:,i]
            j += 1
        cond = cond[:,0,:,:][:,np.newaxis,:,:]
        perm = perm[:,0,:,:][:,np.newaxis,:,:]
    if struct == 'slsimp':
        filename = "gen0/"+struct+nu+"/"
        matx = np.load(filename+struct+nu+"matx.npy")[np.newaxis,5:-5,5:-5,:]
        maty = np.load(filename+struct+nu+"maty.npy")[np.newaxis,5:-5,5:-5,:]
        matz = np.load(filename+struct+nu+"matz.npy")[np.newaxis,5:-5,5:-5,:]
        H1p = np.load(filename+struct+nu+"H1p.npy")[np.newaxis,5:-5,5:-5,:]
        perm = np.load(filename+struct+nu+"permy.npy")[np.newaxis,5:-5,5:-5]
        cond = np.load(filename+struct+nu+"condy.npy")[np.newaxis,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    if struct == 'slsimp90':
        filename = "gen0/"+'slsimp'+nu+"/"
        matx = np.load(filename+'slsimp'+nu+"matx.npy")[np.newaxis]#,5:-5,5:-5,:]
        maty = np.load(filename+'slsimp'+nu+"maty.npy")[np.newaxis]#,5:-5,5:-5,:]
        matz = np.load(filename+'slsimp'+nu+"matz.npy")[np.newaxis]#,5:-5,5:-5,:]
        H1p = np.load(filename+'slsimp'+nu+"H1p.npy")[np.newaxis]#,5:-5,5:-5,:]
        perm = np.load(filename+'slsimp'+nu+"permy.npy")[np.newaxis]#,5:-5,5:-5]
        cond = np.load(filename+'slsimp'+nu+"condy.npy")[np.newaxis]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
        absH1p = np.rot90(absH1p, k = 1, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 1, axes= (-2,-1))
        cond = np.rot90(cond, k = 1, axes= (-2,-1))
        perm = np.rot90(perm, k = 1, axes= (-2,-1))
        matx = np.rot90(matx, k = 1, axes= (-3,-2))
        maty = np.rot90(maty, k = 1, axes= (-3,-2))
        matz = np.rot90(matz, k = 1, axes= (-3,-2))
    if struct == 'slsimp180':
        filename = "gen0/"+'slsimp'+nu+"/"
        matx = np.load(filename+'slsimp'+nu+"matx.npy")[np.newaxis]#,5:-5,5:-5,:]
        maty = np.load(filename+'slsimp'+nu+"maty.npy")[np.newaxis]#,5:-5,5:-5,:]
        matz = np.load(filename+'slsimp'+nu+"matz.npy")[np.newaxis]#,5:-5,5:-5,:]
        H1p = np.load(filename+'slsimp'+nu+"H1p.npy")[np.newaxis]#,5:-5,5:-5,:]
        perm = np.load(filename+'slsimp'+nu+"permy.npy")[np.newaxis]#,5:-5,5:-5]
        cond = np.load(filename+'slsimp'+nu+"condy.npy")[np.newaxis]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
        absH1p = np.rot90(absH1p, k = 2, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 2, axes= (-2,-1))
        cond = np.rot90(cond, k = 2, axes= (-2,-1))
        perm = np.rot90(perm, k = 2, axes= (-2,-1))
        matx = np.rot90(matx, k = 2, axes= (-3,-2))
        maty = np.rot90(maty, k = 2, axes= (-3,-2))
        matz = np.rot90(matz, k = 2, axes= (-3,-2))
    if struct == 'slsimp270':
        filename = "gen0/"+'slsimp'+nu+"/"
        matx = np.load(filename+'slsimp'+nu+"matx.npy")[np.newaxis]#,5:-5,5:-5,:]
        maty = np.load(filename+'slsimp'+nu+"maty.npy")[np.newaxis]#,5:-5,5:-5,:]
        matz = np.load(filename+'slsimp'+nu+"matz.npy")[np.newaxis]#,5:-5,5:-5,:]
        H1p = np.load(filename+'slsimp'+nu+"H1p.npy")[np.newaxis]#,5:-5,5:-5,:]
        perm = np.load(filename+'slsimp'+nu+"permy.npy")[np.newaxis]#,5:-5,5:-5]
        cond = np.load(filename+'slsimp'+nu+"condy.npy")[np.newaxis]#,5:-5,5:-5]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
        absH1p = np.rot90(absH1p, k = 3, axes= (-2,-1))
        phH1p = np.rot90(phH1p, k = 3, axes= (-2,-1))
        cond = np.rot90(cond, k = 3, axes= (-2,-1))
        perm = np.rot90(perm, k = 3, axes= (-2,-1))
        matx = np.rot90(matx, k = 3, axes= (-3,-2))
        maty = np.rot90(maty, k = 3, axes= (-3,-2))
        matz = np.rot90(matz, k = 3, axes= (-3,-2))
    if struct == 'B1_delta':
        filename = struct+"/"+nu+"/"
        matx = np.load(filename+nu+"matx.npy")[np.newaxis,20:-20,20:-20,:]
        maty = np.load(filename+nu+"maty.npy")[np.newaxis,20:-20,20:-20,:]
        matz = np.load(filename+nu+"matz.npy")[np.newaxis,20:-20,20:-20,:]
        H1p = np.load(filename+nu+"H1p.npy")[np.newaxis,20:-20,20:-20,9:12]
        perm = np.load(filename+nu+"permy.npy")[np.newaxis,20:-20,20:-20]
        cond = np.load(filename+nu+"condy.npy")[np.newaxis,20:-20,20:-20]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    if struct == 'B1_delta/v2':
        filename = struct+"/"+nu+"/"
        matx = np.load(filename+nu+"matx.npy")[np.newaxis,33:-37,33:-37,22:26]
        maty = np.load(filename+nu+"maty.npy")[np.newaxis,33:-37,33:-37,22:26]
        matz = np.load(filename+nu+"matz.npy")[np.newaxis,33:-37,33:-37,22:26]
        H1p = np.load(filename+nu+"H1p.npy")[np.newaxis,33:-37,33:-37,22:26]
        H1m = np.load(filename+nu+"H1m.npy")[np.newaxis,33:-37,33:-37,22:26]
        perm = np.load(filename+nu+"permy.npy")[np.newaxis,33:-37,33:-37]
        cond = np.load(filename+nu+"condy.npy")[np.newaxis,33:-37,33:-37]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        phH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            phH1m[0,mm,:,:] = np.angle(H1m[0,:,:,mm],deg=False)
            absH1m[0,mm,:,:] = np.abs(H1m[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    if struct == 'B1_delta/squaring':
        filename = struct+"/"+nu
        matx = np.load(filename+"matx.npy")[np.newaxis,30:-34,30:-34,22:26]
        maty = np.load(filename+"maty.npy")[np.newaxis,30:-34,30:-34,22:26]
        matz = np.load(filename+"matz.npy")[np.newaxis,30:-34,30:-34,22:26]
        H1p = np.load(filename+"H1p.npy")[np.newaxis,30:-34,30:-34,22:26]
        H1m = np.load(filename+"H1m.npy")[np.newaxis,30:-34,30:-34,22:26]
        perm = np.load(filename+"permy.npy")[np.newaxis,30:-34,30:-34]
        cond = np.load(filename+"condy.npy")[np.newaxis,30:-34,30:-34]
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        phH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            phH1m[0,mm,:,:] = np.angle(H1m[0,:,:,mm],deg=False)
            absH1m[0,mm,:,:] = np.abs(H1m[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    if struct == 'PINN_model':
        filename = struct+"/"+nu+'/'+nu
        matx = np.load(filename+"matx.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        maty = np.load(filename+"maty.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        matz = np.load(filename+"matz.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        H1p = np.load(filename+"H1p.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        H1m = np.load(filename+"H1m.npy")[np.newaxis,4:-3,4:-4,0:15]#,3:-2,3:-3,0:15]#
        perm = np.load(filename+"permy.npy")[np.newaxis,4:-3,4:-4]#3:-2,3:-3]#
        cond = np.load(filename+"condy.npy")[np.newaxis,4:-3,4:-4]#3:-2,3:-3]#
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        phH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            phH1m[0,mm,:,:] = np.angle(H1m[0,:,:,mm],deg=False)
            absH1m[0,mm,:,:] = np.abs(H1m[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    if struct == 'Tumor_sizes':
        filename = struct+"/"+nu+'/'+nu
        matx = np.load(filename+"matx.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        maty = np.load(filename+"maty.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        matz = np.load(filename+"matz.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        H1p = np.load(filename+"H1p.npy")[np.newaxis,4:-3,4:-4,0:15]#3:-2,3:-3,0:15]#
        H1m = np.load(filename+"H1m.npy")[np.newaxis,4:-3,4:-4,0:15]#,3:-2,3:-3,0:15]#
        perm = np.load(filename+"permy.npy")[np.newaxis,4:-3,4:-4]#3:-2,3:-3]#
        cond = np.load(filename+"condy.npy")[np.newaxis,4:-3,4:-4]#3:-2,3:-3]#
        phH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1p = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        phH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        absH1m = np.zeros((1,H1p.shape[3],H1p.shape[1],H1p.shape[2]))
        for mm in range(2):
            phH1p[0,mm,:,:] = np.unwrap(np.angle(H1p[0,:,:,mm],deg=False))
            absH1p[0,mm,:,:] = np.abs(H1p[0,:,:,mm])
            phH1m[0,mm,:,:] = np.angle(H1m[0,:,:,mm],deg=False)
            absH1m[0,mm,:,:] = np.abs(H1m[0,:,:,mm])
        cond = cond[:,:,:,0][:,np.newaxis,:,:]
        perm = perm[:,:,:,0][:,np.newaxis,:,:]
    return matx, maty, matz, phH1p, absH1p, cond, perm#, phH1m, absH1m