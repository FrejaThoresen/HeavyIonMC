
import random

from scipy import stats
import numpy as np
import flowfact
from flowfact.utils import wrap_2pi
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from scipy.stats import cosine

from scipy.special import gamma

def event_gen_flow(nparts, nevents):
    """
    Generator for pure-flow events. Defined elsewhere, but just a simple cosine distribution
    """
    v2 = 0.2
    flow = flowfact.Flow_gen([0, 0.2], psi=[0, 0])
    parts, v, psi =flow.rvs_with_parameters(size=[nevents, nparts]) 
    for phis in parts:
        uni_etas = stats.uniform.rvs(0,2, nparts) -1
        #print 'psi', psi
        #print 'flow', v
        yield np.stack([uni_etas, phis], axis=1) # wrap_2pi(phis-psi[1])
        

def makeBinnedParticles(parts, xbins, ybins):
    """
    Particles binned in eta and phi
    """
    hist, x_edges, y_edges = np.histogram2d(parts[:, 0], parts[:, 1], [xbins, ybins])

    y_centers = y_edges + (y_edges[1] - y_edges[0])/2
    x_centers = x_edges + (x_edges[1] - x_edges[0])/2

    X = x_centers[0:-1]
    Y = y_centers[0:-1]

    X = np.repeat(X,xbins)
    Y = np.tile(Y,ybins)

    comb = np.array([X,Y])
    comb = comb.T
    w =  hist.reshape(1,xbins*ybins)
    return comb, w


def corr2(phis_a, phis_b, n, same_sub):
    """2-particle correlations"""

    _all = np.cos(n*(phis_a[0,:, None] - phis_b[0,None, :]))

    if same_sub:
        # indices to the lower triangle in the phi2-phi2 plane to avoid doule counting
        tril_idx1, tril_idx2 = np.tril_indices_from(_all, k=-1)
        return np.mean(_all[tril_idx1, tril_idx2, ...])
    else:
        # avoid double the upper triangle but keep the diagonal
        tril_idx1, tril_idx2 = np.tril_indices_from(_all, k=0)
        return np.mean(_all[tril_idx1, tril_idx2, ...])

def corr2_GF(particles, w, n):
    """n-particle correlation using the Generic Framework (GF)"""
    Q_vec = {'real' : 0.0, 'imaginary': 0.0, 'weight sq.' :0.0, 'sum weight sq.' : 0.0}
    #print w[:,0].shape
    #print particles[:,1].shape
    Q_vec['real'] = np.sum(np.power(w[:],1)*np.cos(2*particles[:,1]))
    Q_vec['imaginary'] = np.sum(np.power(w[:],1)*np.sin(2*particles[:,1]))
    Q_vec['weight sq.'] = np.sum(np.power(w[:],2))
    Q_vec['sum weight'] = np.sum(np.power(w[:],1))

    nn2 = np.power(Q_vec['real'],2) + np.power(Q_vec['imaginary'],2) - Q_vec['weight sq.']
    dn2 = np.power(Q_vec['sum weight'],2) - Q_vec['weight sq.']
    #print 'Q', Q_vec
    #print 'nn2 = ', nn2
    #print 'dn2 = ', dn2
    two = {'nn2' : nn2, 'dn2' : dn2}

    return two
    
def calc_c22(args):
    """
    Generate an event and compute c_2{2} for it.
    This function used the pure-flow PDF. Used as a cross-check
    """
    nparts, nevents = args
    
    nn2_smear = []
    dn2_smear = []
    two_true = []
    nn2_true = []
    dn2_true = []

    nn2_smear_w = []
    dn2_smear_w = []    
    nn2 = []
    dn2 = []
    nn2_binned = []
    dn2_binned = []    
    nn2_binned_noweights = []
    dn2_binned_noweights = []        
    nn2_binned_corr = []
    dn2_binned_corr = []  
    auto = []
    for parts in event_gen_flow(nparts, nevents):
        weights = np.full_like(parts[:,0],1.)
        two = corr2_GF(parts,weights,2)
        nn2.append(two['nn2'])
        dn2.append(two['dn2'])
        
        index = np.where((parts[:,1] > np.pi)) #(parts[:,0] > 0) & 
        weights[index] = 0.4

        for i in range(0,len(weights[:])):
            if (random.random() > weights[i]):
                parts[i,0] = -2.0
        
        particles = parts[np.logical_and(-1 <= parts[:, 0], parts[:, 0] < 1)]
        w = np.full_like(particles[:,0],1.)
        two_smear = corr2_GF(particles,w,2)
        nn2_smear.append(two_smear['nn2'])
        dn2_smear.append(two_smear['dn2'])
        
        new_index = np.where((particles[:,0] > 0) & (particles[:,1] > np.pi))
        w[new_index] = 1.0/0.4
        two_smear_w = corr2_GF(particles,w,2)
        nn2_smear_w.append(two_smear_w['nn2'])
        dn2_smear_w.append(two_smear_w['dn2'])
        
        particles, w = makeBinnedParticles_noweights(particles,20,20)
        w = np.full_like(particles[:,1],1)
        two_binned_noweights = corr2_GF(particles,w,2)
        nn2_binned_noweights.append(two_binned_noweights['nn2'])
        dn2_binned_noweights.append(two_binned_noweights['dn2'])  
        
        particles, w = makeBinnedParticles(particles,20,20)
        w = w[0]
        two_binned = corr2_GF(particles,w,2)
        nn2_binned.append(two_binned['nn2'])
        dn2_binned.append(two_binned['dn2'])        
        
        for i in range(0,len(w)):
            if (w[i]>0.0):
                auto.append(np.sum(gamma(w[i]+1)/gamma(w[i]-1)))

    c2 = {'nn2' : np.sum(nn2), 'dn2' : np.sum(dn2), 'auto' : 0.0}
    c2_binned = {'nn2' : np.sum(nn2_binned), 'dn2' : np.sum(dn2_binned), 'auto' : np.sum(auto)}
    c2_smear = {'nn2' : np.sum(nn2_smear_w), 'dn2' : np.sum(dn2_smear_w), 'auto' : 0.0}
    c2_smear_w = {'nn2' : np.sum(nn2_smear_w), 'dn2' : np.sum(dn2_smear_w), 'auto' : 0.0}
    c2_binned_noweights = {'nn2' : np.sum(nn2_binned_noweights), 'dn2' : np.sum(dn2_binned_noweights), 'auto' : 0.0}
    
    return c2, c2_smear, c2_smear_w, c2_binned_noweights, c2_binned

def makeBinnedParticles_noweights(parts, xbins, ybins):
    """
    Bin particles in eta and phi, but do not use multiplicity weights
    """
    w = np.full_like(parts[:,0],1.)

    hist, x_edges, y_edges = np.histogram2d(parts[:, 0], parts[:, 1], [xbins, ybins])
    y_centers = y_edges + (y_edges[1] - y_edges[0])/2
    x_centers = x_edges + (x_edges[1] - x_edges[0])/2

    X = x_centers[0:-1]
    Y = y_centers[0:-1]

    X = np.repeat(X,xbins)
    Y = np.tile(Y,ybins)

    comb = np.array([X,Y])
    comb = comb.T
    w =  hist.reshape(1,xbins*ybins)

    comb = comb.tolist()
    comb1 = []
    for i in range(0,len(w[0])):
        j = 0
        while (w[0][i] > j):
            
            comb1.append(comb[i])
            j = j +1
    w = np.full_like(w[0],1.0)
    comb = np.array(comb1)
    return comb, w
    
def plotEta(parts,w):
    """1D plotting script in eta"""
    count, bins, ignored = plt.hist(parts[:,1], 50, density=True,label='drawn values',alpha=0.7,weights=w)
    #plt.plot(np.linspace(0,2*np.pi,len(parts[:,1])),parts[:,1],'o')
    plt.xlabel(r'$\varphi$',fontsize=12)
    plt.ylabel(r'$\frac{dN}{d\varphi}$',fontsize=14)
    y = 1.0/2/np.pi * (1 + (2*0.2*np.cos(2*(bins))))
    v = np.mean(2*np.cos(2*bins))

    two = corr2_GF(parts, w, 2)

    v_calc = np.power(two['nn2']/two['dn2'],0.5)
    y_calc = 1.0/2/np.pi * (1 + (2*v_calc*np.cos(2*(bins))))

    plt.plot(bins, y,linewidth=2,label='pdf with $v_2 = 0.2$')
    plt.plot(bins, y_calc,linewidth=2, label='calc. $v_2 = $' + '%.2f' %(v_calc))
    plt.legend(fontsize=12)
    plt.gcf().text(0.15,0.15,'1000 particles',fontsize=12)
    plt.xlim([0,2*np.pi])
    plt.savefig('dNdphi.pdf',bbox_inches='tight')

    plt.show()
    
def plotPhi(parts,w):
    """1D plotting script in phi, NEED FIX FOR SHIFT"""

    count, bins, ignored = plt.hist(parts[:,0], 50, density=True,label='drawn values',alpha=0.7,weights=w)
    plt.plot(bins, stats.uniform.pdf(bins+1,scale=2),label='uniform pdf',linewidth=2)
    plt.xlabel(r'$\eta$',fontsize=12)
    plt.gcf().text(0.15,0.15,'1000 particles',fontsize=12)

    plt.ylabel(r'$\frac{dN}{d\eta}$',fontsize=14)
    plt.xlim([-1,1])
    plt.legend(fontsize=12)
    plt.savefig('dNdeta.pdf',bbox_inches='tight')
    plt.show()    