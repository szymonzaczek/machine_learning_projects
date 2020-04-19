import sys

print(sys.version)

import mdtraj as md
import os
import pyemma
import re
import os
import string
import matplotlib.pyplot as plt
import numpy as np
import pyemma.msm as msm
import pyemma.plots as mplt
import pyemma.coordinates as coor
from time import perf_counter

a1 = perf_counter()
# Selecting 'CA' atoms from the active site - md_traj part
mdtraj_traj = md.load('test_size.nc', top='stripped_adduct.pdb')
mdtraj_top = mdtraj_traj.topology
as_list = [84,85,86,89,90,91,92,93,114,115,116,117,118,119,121,122,125,126,129,130,376,378,379,390,394,409,410,432,433,434,435,459,460,461,462,464,465,466,468,469,522,523,524,525,526,527,528,530,538,541,558,559,560,561,562,567]
atoms_list = [atom.index for atom in mdtraj_top.atoms if atom.name == 'CA' if int(str(atom.residue)[3:]) in as_list]

# PyEmma Part - TICA, clustering and MSM
top = 'stripped_adduct.prmtop'
trajs = ('test_size.nc')

feat = coor.featurizer(top)
feat.add_contacts(indices=atoms_list, threshold=1)
print(feat.dimension())

# Conducting TICA
inp = coor.source(trajs, feat)
t0 = perf_counter()
tica_obj = coor.tica(inp, lag=500, var_cutoff=0.9, kinetic_map=True, stride=1)
t1 = perf_counter()
print(f"Time elapsed: {round(t1-t0)} s")

print(dir(tica_obj.dim))

print(tica_obj.dimension())
print(len(tica_obj.cumvar))



# testing different time lags
#for tica_obj in tica_objects:
#    #print(sum(tica_obj.cumvar[:10]))
#    Y = tica_obj.get_output()
#    plt.figure(figsize=(8,5))
#    ax1=plt.subplot(311)
#    x = dt*np.arange(Y[0].shape[0])
#    plt.plot(x, Y[0][:,0]); plt.ylabel('IC 1'); plt.xticks([]); plt.yticks(np.arange(-2, 3))
#    ax1=plt.subplot(312)
#    plt.plot(x, Y[0][:,1]); plt.ylabel('IC 2'); plt.xticks([]);  plt.yticks(np.arange(-2, 4))
#    ax1=plt.subplot(313)
#    plt.plot(x, Y[0][:,2]); plt.xlabel('time / ns'); plt.ylabel('IC 3'); plt.yticks(np.arange(-4, 6, 2))   

dt = 0.005
Y = tica_obj.get_output()
xall = np.vstack(Y)[:,0]
yall = np.vstack(Y)[:,1]


plt.figure(figsize=(8,5))
ax1=plt.subplot(311)
x = dt*np.arange(Y[0].shape[0])
plt.plot(x, Y[0][:,0]); plt.ylabel('IC 1'); plt.xticks([]); plt.yticks(np.arange(-2, 3))
ax1=plt.subplot(312)
plt.plot(x, Y[0][:,1]); plt.ylabel('IC 2'); plt.xticks([]);  plt.yticks(np.arange(-2, 4))
ax1=plt.subplot(313)
plt.plot(x, Y[0][:,2]); plt.xlabel('time / ns'); plt.ylabel('IC 3'); plt.yticks(np.arange(-4, 6, 2)) 

# for shorter trajectory, ideal number of clusters is 100
# optimal lag_time = 750?
    

# optimal lag_time = 1000 timesteps

clustering = coor.cluster_kmeans(Y, k=100)
dtrajs = clustering.dtrajs    
msm = pyemma.msm.estimate_markov_model(dtrajs, 380)
pyemma.plots.plot_cktest(msm.cktest(3, err_est=True), marker='.')

# TRIALS - reg_space clustering and kmeans comparison - kmeans by far better
clustering_reg = coor.cluster_regspace(Y, dmin=2, max_centers=100)
cr_x = clustering_reg.clustercenters[:,0]
cr_y = clustering_reg.clustercenters[:,0]
cc_x = clustering.clustercenters[:,0]
cc_y = clustering.clustercenters[:,1]
c_reg = [cr_x, cr_y]
c = [cc_x, cc_y]
print(len(clustering_reg.clustercenters))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, cls in zip(axes.flat, [c, c_reg]):
    pyemma.plots.plot_density(xall, yall, ax=ax, cbar=False, alpha=0.1, logscale=True)
    ax.scatter(*cls)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
fig.tight_layout()

# comparing feature histograms
vamp = coor.vamp(inp, lag=500, stride=1)
vamp_output = vamp.get_output()
tica_concatenated = Y[0]
tica_concatenated = tica_concatenated[:,:5]
vamp_concatenated = vamp_output[0]
vamp_concatenated = vamp_concatenated[:,:5]

pyemma.plots.plot_feature_histograms(vamp_concatenated)
pyemma.plots.plot_feature_histograms(tica_concatenated)


# comparing how feature values differ - theyre somehow similar, though VAMP changes more; plateuos should be looked for here
tica_concatenated = Y[0]
tica_concatenated = tica_concatenated[:,0]
vamp_concatenated = vamp_output[0]
vamp_concatenated = vamp_concatenated[:,0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tica_concatenated[:300], label='TICA')
# note that for better comparability, we enforce the same direction as TICA
ax.plot(vamp_concatenated[:300] * -1, label='VAMP')
ax.set_xlabel('time / steps')
ax.set_ylabel('feature values')
ax.legend()
fig.tight_layout()

tica_concatenated = Y[0]
tica_concatenated = tica_concatenated[:,1]
vamp_concatenated = vamp_output[0]
vamp_concatenated = vamp_concatenated[:,1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tica_concatenated[:300], label='TICA')
# note that for better comparability, we enforce the same direction as TICA
ax.plot(vamp_concatenated[:300] * -1, label='VAMP')
ax.set_xlabel('time / steps')
ax.set_ylabel('feature values')
ax.legend()
fig.tight_layout()

tica_concatenated = Y[0]
tica_concatenated = tica_concatenated[:,2]
vamp_concatenated = vamp_output[0]
vamp_concatenated = vamp_concatenated[:,2]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tica_concatenated[:300], label='TICA')
# note that for better comparability, we enforce the same direction as TICA
ax.plot(vamp_concatenated[:300] * -1, label='VAMP')
ax.set_xlabel('time / steps')
ax.set_ylabel('feature values')
ax.legend()
fig.tight_layout()


# density plots - they show that tica and vamp are practically the same
tica_concatenated = Y[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[0], cbar=False)
pyemma.plots.plot_free_energy(*tica_concatenated[:, :2].T, ax=axes[1], legacy=False)

vamp_concatenated = vamp_output[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pyemma.plots.plot_density(*vamp_concatenated[:, :2].T* -1, ax=axes[0], cbar=False)
pyemma.plots.plot_free_energy(*vamp_concatenated[:, :2].T* -1, ax=axes[1], legacy=False)

# TRIALS END HERE


# optimal parameters according to cktest: n_clusters = 100, lag_time = 380;
# optimal numbers of dtrajs parsed to cktest: 3, 4, 5, 7; perhaps 5 is the best

print(Y)

print((len(Y)))
# xall, yall - first and second tica output dimensions

xall = np.vstack(Y)[:,0]
yall = np.vstack(Y)[:,1]
zall = np.vstack(Y)[:,2]
W = np.concatenate(msm.trajectory_weights())

mplt.plot_free_energy(xall, yall)
cc_x = clustering.clustercenters[:,0]
cc_y = clustering.clustercenters[:,1]
plt.plot(cc_x,cc_y, linewidth=0, marker='o', markersize=5, color='black')

print('fraction of states used = {:f}'.format(msm.active_state_fraction))
print('fraction of counts used = {:f}'.format(msm.active_count_fraction))

print(msm.stationary_distribution)
print('sum of weights = {:f}'.format(msm.pi.sum()))

print(dir(msm))
print(dir(tica_obj))

# stationary distribution
fig, ax, misc = pyemma.plots.plot_contour(
    xall, yall, msm.pi[clustering.dtrajs[0]],
    cbar_label='stationary distribution',
    method='nearest', mask=True)
ax.scatter(cc_x,cc_y, s=15, c='C1')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal')
fig.tight_layout()

# corrected free energy plot with stationary distribution

fig, ax, misc = pyemma.plots.plot_free_energy(
    xall, yall,
    weights=np.concatenate(msm.trajectory_weights()),
    legacy=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal')
fig.tight_layout()

eigvec = msm.eigenvectors_right()

# first eigenvector is the slowest process; in this case this is a transition from the righ
# this plot should be used to verify how many metastable states should be used - in this case, perhaps 2 are enough; third one is not anyhow separated by tica; 4th and 5th have more sense
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
for i, ax in enumerate(axes.flat):
    pyemma.plots.plot_contour(
        xall, yall, eigvec[clustering.dtrajs[0], i + 1], ax=ax, cmap='PiYG',
        cbar_label='{}. right eigenvector'.format(i + 2), mask=True)
    ax.scatter(cc_x,cc_y, s=15, c='C1')
    ax.set_xlabel('$x$')
    ax.set_aspect('equal')
axes[0].set_ylabel('$y$')
fig.tight_layout()

# which states are relevant might be taken from eigenvectors a

print(eigvec)
print('first eigenvector is one: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 0], 1, atol=1e-15), eigvec[:, 0].min(), eigvec[:, 0].max()))

print('second eigenvector: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 1], 1, atol=1e-15), eigvec[:, 1].min(), eigvec[:, 1].max()))

print('third eigenvector: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 2], 1, atol=1e-15), eigvec[:, 2].min(), eigvec[:, 2].max()))

print('fourth eigenvector: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 3], 1, atol=1e-15), eigvec[:, 3].min(), eigvec[:, 3].max()))

print('fifth eigenvector: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 4], 1, atol=1e-15), eigvec[:, 4].min(), eigvec[:, 4].max()))

print('sixth eigenvector: {} (min={}, max={})'.format(
    np.allclose(eigvec[:, 5], 1, atol=1e-15), eigvec[:, 5].min(), eigvec[:, 5].max()))

# Five eigenvectors should be enough, sixth one introduces big changes in min - max values

#nstates = 5
nstates=3
msm.pcca(nstates)


for i, s in enumerate(msm.metastable_sets):
    print('π_{} = {:f}'.format(i + 1, msm.pi[s].sum()))

mfpt = np.zeros((nstates, nstates))
for i in range(nstates):
    for j in range(nstates):
        mfpt[i, j] = msm.mfpt(
            msm.metastable_sets[i],
            msm.metastable_sets[j])

from pandas import DataFrame
print('MFPT / steps:')
DataFrame(np.round(mfpt, decimals=2), index=range(1, nstates + 1), columns=range(1, nstates + 1))

bayesian_msm = pyemma.msm.bayesian_markov_model(dtrajs, lag=380, conf=0.95)


mfpt_sample = np.zeros((nstates, nstates, bayesian_msm.nsamples))
for i in range(nstates):
    for j in range(nstates):
        mfpt_sample[i, j] = bayesian_msm.sample_f(
            'mfpt',
            msm.metastable_sets[i],
            msm.metastable_sets[j])

fig, ax = plt.subplots()
ax.hist(mfpt_sample[0, 1], histtype='step', label='MS 1 -> MS 2', density=True)
ax.hist(mfpt_sample[0, 2], histtype='step', label='MS 1 -> MS 3', density=True)
ax.hist(mfpt_sample[1, 0], histtype='step', label='MS 2 -> MS 1', density=True)
ax.hist(mfpt_sample[1, 2], histtype='step', label='MS 2 -> MS 3', density=True)
ax.hist(mfpt_sample[2, 0], histtype='step', label='MS 3 -> MS 1', density=True)
ax.hist(mfpt_sample[2, 1], histtype='step', label='MS 3 -> MS 2', density=True)
ax.set_xlabel('MFPT (steps)')
ax.set_title('Bayesian MFPT sample histograms')
fig.legend()#loc=10)

# tpt between two first metastable sets
A = msm.metastable_sets[0]
B = msm.metastable_sets[1]
C = msm.metastable_sets[2]
#D = msm.metastable_sets[3]
#E = msm.metastable_sets[4]
flux = pyemma.msm.tpt(msm, A, B)
# PREVIOUS ANALYSIS ENDED HERE

print(f"A: {A}, B: {B}, C: {C}")
print(flux)

# CAVEAT: incorporate cluster centers here
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes.flat):
    pyemma.plots.plot_contour(
        xall, yall, msm.metastable_distributions[i][dtrajs[0]], ax=ax, cmap='afmhot_r', 
        mask=True, method='nearest', cbar_label='metastable distribution {}'.format(i + 1))

fig, axes = plt.subplots(1, 3, figsize=(20, 3))
for i, ax in enumerate(axes.flat):
    pyemma.plots.plot_contour(
        xall, yall,
        msm.metastable_distributions[i][dtrajs[0]],
        ax=ax,
        cmap='afmhot_r', 
        mask=True,
        cbar_label='metastable distribution {}'.format(i + 1))
    ax.set_xlabel('$\Phi$')
axes[0].set_ylabel('$\Psi$')
fig.tight_layout()


cc_x = clustering.clustercenters[:,0]
cc_y = clustering.clustercenters[:,1]

print(dir(msm))
print(msm.timescales)


# FIXME: 3 states have the same highest membership - this should not happen
metastable_traj = msm.metastable_assignments[dtrajs]
highest_membership = msm.metastable_distributions.argmax(1)
coarse_state_centers = clustering.clustercenters[msm.active_set[highest_membership]]



# CAVEAT: 3 states have the same course_state center in first two dimensions - try to apply pca to them
test = coarse_state_centers[:,0]
test2 = coarse_state_centers[:,1]
test3 = np.column_stack((test, test2))


mfpt = np.zeros((nstates, nstates))
for i in range(nstates):
    for j in range(nstates):
        mfpt[i, j] = msm.mfpt(
            msm.metastable_sets[i],
            msm.metastable_sets[j])

inverse_mfpt = np.zeros_like(mfpt)
nz = mfpt.nonzero()
inverse_mfpt[nz] = 1.0 / mfpt[nz]  


# best plot so far
fig, ax = plt.subplots(figsize=(10, 7))
_, _, misc = pyemma.plots.plot_state_map(
    xall, yall, metastable_traj, ax=ax, zorder=-1)
misc['cbar'].set_ticklabels(range(1, nstates + 1))  # set state numbers 1 ... nstates
pyemma.plots.plot_network(
    inverse_mfpt,
    pos=coarse_state_centers,
    figpadding=1,
    arrow_label_format='%.1f ps',
    arrow_labels=mfpt,
    size=15,
    show_frame=True,
    ax=ax)

# automatic assignments of state positions
fig, ax = plt.subplots(figsize=(10, 7))
_, _, misc = pyemma.plots.plot_state_map(
    xall, yall, metastable_traj, ax=ax, zorder=-1)
misc['cbar'].set_ticklabels(range(1, nstates + 1))  # set state numbers 1 ... nstates
pyemma.plots.plot_network(
    inverse_mfpt,
    pos=coarse_state_centers,
    figpadding=1,
    #rrow_label_format='%.1f ps',
    arrow_labels=mfpt,
    size=12,
    show_frame=True,
    ax=ax)
    

# just the network, without mapping onto coordinates

fig, ax = plt.subplots(figsize=(10, 7)) 
pyemma.plots.plot_network(
    inverse_mfpt,
    #pos=test3,
    figpadding=0,
    #rrow_label_format='%.1f ps',
    arrow_labels=mfpt,
    size=12,
    show_frame=True,
    ax=ax)
    
# GREAT PLOT
flux = pyemma.msm.tpt(msm, A, B)
cg, cgflux = flux.coarse_grain(msm.metastable_sets)
fig, ax = plt.subplots(figsize=(10, 7))
pyemma.plots.plot_contour(
    xall,yall,
    flux.committor[clustering.dtrajs],
    cmap='brg',
    ax=ax,
    mask=True,
    cbar_label=r'committor 1 $\to$ 4',
    alpha=0.8,
    zorder=-1);
pyemma.plots.plot_flux(
    cgflux,
    coarse_state_centers,
    cgflux.stationary_distribution,
    state_labels=['A','B', 'C'], 
    ax=ax,
    show_committor=False,
    figpadding=1,
    show_frame=True,
    size=20,
    arrow_label_format='%2.e / ps');
        
paths, path_fluxes = cgflux.pathways(fraction=0.99)
print('percentage       \tpath')
print('-------------------------------------')
for i in range(len(paths)):
    print(np.round(path_fluxes[i] / np.sum(path_fluxes), 3),' \t', paths[i] + 1)




























def visualize_metastable(samples, cmap, selection='backbone'):
    """ visualize metastable states
    Parameters
    ----------
    samples: list of mdtraj.Trajectory objects
        each element contains all samples for one metastable state.
    cmap: matplotlib.colors.ListedColormap
        color map used to visualize metastable states before.
    selection: str
        which part of the molecule to selection for visualization. For details have a look here:
        http://mdtraj.org/latest/examples/atom-selection.html#Atom-Selection-Language
    """
    import nglview
    from matplotlib.colors import to_hex

    widget = nglview.NGLWidget()
    widget.clear_representations()
    ref = samples[0]
    for i, s in enumerate(samples):
        s = s.superpose(ref)
        s = s.atom_slice(s.top.select(selection))
        comp = widget.add_trajectory(s)
        comp.add_ball_and_stick()

    # this has to be done in a separate loop for whatever reason...
    x = np.linspace(0, 1, num=len(samples))
    for i, x_ in enumerate(x):
        c = to_hex(cmap(x_))
        widget.update_ball_and_stick(color=c, component=i, repr_index=i)
        widget.remove_cartoon(component=i)
    return widget


# dont know if this works
import matplotlib as mpl
cmap = mpl.cm.get_cmap('viridis', nstates)
my_samples = [pyemma.coordinates.save_traj(inp, idist, outfile=None, top=top)
              for idist in msm.sample_by_distributions(msm.metastable_distributions, 50)]

visualize_metastable(my_samples, cmap, selection='backbone')
# until here
    

flux_1_3 = pyemma.msm.tpt(msm, A, C)
cg, cgflux = flux_1_3.coarse_grain(msm.metastable_sets)
fig, ax = plt.subplots(figsize=(10, 7))

pyemma.plots.plot_contour(
    xall, yall,
    flux_1_3.committor[dtrajs],
    cmap='brg',
    ax=ax,
    mask=True,
    cbar_label=r'committor 1 $\to$ 4',
    alpha=0.8,
    zorder=-1);

pyemma.plots.plot_flux(
    cgflux,
    test3,
    cgflux.stationary_distribution,
    state_labels=['A','' ,'', 'B', ''], 
    ax=ax,
    show_committor=False,
    figpadding=0,
    show_frame=True,
    arrow_label_format='%2.e / ps');
    
    
    
    

plt.plot(tica_obj.timescales)

print('TICA dimensions: ', tica_obj.dimension())

for i in range(2):
    if tica_obj.eigenvectors[0, i] > 0: 
        tica_obj.eigenvectors[:, i] *= -1
print(tica_obj.cumvar[:20])

Y = tica_obj.get_output()
mplt.plot_free_energy(np.vstack(Y)[:,0], np.vstack(Y)[:,1])
print(Y)

dt = 0.005
plt.figure(figsize=(8,5))
ax1=plt.subplot(311)
x = dt*np.arange(Y[0].shape[0])
plt.plot(x, Y[0][:,0]); plt.ylabel('IC 1'); plt.xticks([]); plt.yticks(np.arange(-2, 3))
ax1=plt.subplot(312)
plt.plot(x, Y[0][:,1]); plt.ylabel('IC 2'); plt.xticks([]);  plt.yticks(np.arange(-2, 4))
ax1=plt.subplot(313)
plt.plot(x, Y[0][:,2]); plt.xlabel('time / ns'); plt.ylabel('IC 3'); plt.yticks(np.arange(-4, 6, 2))

n_clusters = 50
clustering = coor.cluster_kmeans(Y,k=n_clusters)
dtrajs = clustering.dtrajs

mplt.plot_free_energy(np.vstack(Y)[:,0], np.vstack(Y)[:,1])
cc_x = clustering.clustercenters[:,0]
cc_y = clustering.clustercenters[:,1]
plt.plot(cc_x,cc_y, linewidth=0, marker='o', markersize=5, color='black')

lags = [1,5,10,20,35,50,75,100,150,200,300,400,500,600,700,800,900,100]

implied_ts = pyemma.msm.its(dtrajs=dtrajs,lags=lags, nits=5)
pyemma.plots.plot_implied_timescales(implied_ts,units='time-steps', ylog=False)
#plt.vlines(2,ymin=0,ymax=350,linestyles='dashed')
#plt.annotate("selected model", xy=(lags[-3], implied_ts.timescales[-3][0]), xytext=(15,250),
#                 arrowprops=dict(facecolor='black', shrink=0.001, width=0.1,headwidth=8))
plt.figure(figsize=(10,10),dpi=600)
plt.ylim([0,150])

print(implied_ts)





its = msm.timescales_msm(dtrajs, lags=50, nits=10)
print(its)
mplt.plot_implied_timescales(its, ylog=False, units='steps', linewidth=2)
#plt.xlim(0, 40); plt.ylim(0, 50)

its = msm.timescales_msm(dtrajs, lags=50, nits=10, errors='bayes', n_jobs=-1)
plt.figure(figsize=(8,5))
mplt.plot_implied_timescales(its, show_mean=False, ylog=False, dt=0.1, units='ns', linewidth=2)
#plt.xlim(0, 5); plt.ylim(0.1,60);