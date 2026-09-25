"""Verbatim original Tempotron and spike grouping; no Brian2/MPI imports.
Source: Analysis_CoupledCri_2026/Cluster_Programe/MPI_GlobalBalance_2026.py.
The 50..149 ms voltage grid matches the subsample spike data.
"""
import numpy as np

def Tempotron(SPIKETRAIN0,SPIKETRAIN1):
    def VmaxANDtmaxfun(W, spiketrain0):
        # W is the Nx1 vector; spiketrain0 is the 800 list, each value is the spiketrain of a signal neuron
        V = np.zeros(100)
        t = np.linspace(50, 150 - 1, 100)
        for neuron in range(0, 1600):
            for tspike in spiketrain0[neuron]:
                V = V + W[neuron] * Synaptic_activition(t - tspike)
        V_max = np.max(V)
        t_max = t[np.argmax(V)]
        return t_max, V_max, V

    def delta_Wfun(t_max, spiketrain0):
        delta_W = np.zeros((1600))
        for neuron in range(0, 1600):
            spiketimelist = np.array(spiketrain0[neuron])
            spiketimelist[spiketimelist > t_max] = t_max
            delta_W[neuron] = np.sum(Synaptic_activition(t_max - spiketimelist))
        return delta_W

    def Synaptic_activition(delta_t):
        tau_r, tau_de = 0.5, 2
        delta_t[delta_t < 0] = 0
        return -np.exp(-delta_t / tau_r) + np.exp(-delta_t / tau_de)

    trail_train = np.arange(0, 20)
     # Goal is to let the W readout across 1 for spiketrain0, and keep less than 1 for spiketrain1
    Lambda = 0.5  # learning rate
    V_threshold = 10
    Maxstep = 400
    # W = 0.1*np.ones((800)) #initial weight
    W = np.abs(np.random.normal(0,0.1,1600))
    W_rec = np.zeros((1600,Maxstep))
    for i in range(0,Maxstep):
        delta_W = 0
        # calculate t_max from V
        V_max_pos,V_max_neg = 0,0
        V_max_poslist,V_max_neglist=[],[]
        for trail in trail_train:
            t_max_pos,V_max_pos0,_ = VmaxANDtmaxfun(W,SPIKETRAIN0[trail])
            t_max_neg,V_max_neg0,_ = VmaxANDtmaxfun(W,SPIKETRAIN1[trail])# Calculate \delta W according to gradient-based learning rule
            delta_W = delta_W + Lambda*(delta_Wfun(t_max_pos,SPIKETRAIN0[trail])-
                              delta_Wfun(t_max_neg,SPIKETRAIN1[trail]))/len(trail_train)
            V_max_neg,V_max_pos = V_max_neg+V_max_neg0/len(trail_train),V_max_pos+V_max_pos0/len(trail_train)
            V_max_poslist.append(V_max_pos0)
            V_max_neglist.append(V_max_neg0)

        print('Here is step' + str(i) + ';V_max_pos' + str(min(V_max_poslist)) + ';V_max_neg' + str(max(V_max_neglist)))
        W = W + delta_W
        W_rec[:, i] = W

        condition_pass = ((max(V_max_neglist) < V_threshold) and (min(V_max_poslist) > V_threshold))
        condition_quit = ((max(V_max_neglist) > V_threshold*2) and (min(V_max_poslist) > V_threshold*2))

        if condition_pass or condition_quit:
            break
    V_max_poslist,V_max_neglist = [],[]

    V_max_poslist, V_max_neglist, t_temptornwork = [], [], np.zeros((100))
    for trail in range(20, 100):
        # print('Here is test' + str(trail) + '/100')
        t_temptornwork[trail], V_max_pos_prd, _ = VmaxANDtmaxfun(W, SPIKETRAIN0[trail])
        V_max_poslist.append(V_max_pos_prd)
        _, V_max_neg_prd, _ = VmaxANDtmaxfun(W, SPIKETRAIN1[trail])
        V_max_neglist.append(V_max_neg_prd)

    V_max_neglist, V_max_poslist = np.array(V_max_neglist), np.array(V_max_poslist)
    CR = (len(np.where(V_max_neglist > V_threshold)[0]) + len(np.where(V_max_poslist < V_threshold)[0])) / (160)

    data={'acc':1-CR,'weight':W,'i':i,'t_temptron':t_temptornwork,
          'V_max_neglist':V_max_neglist,'V_max_poslist':V_max_poslist}

    return 1-CR,W,i,t_temptornwork

def Synaptic_activition(delta_t):
    tau_r, tau_de = 0.5, 2
    delta_t[delta_t < 0] = 0
    return -np.exp(-delta_t / tau_r) + np.exp(-delta_t / tau_de)

def VmaxANDtmaxfun(W, spiketrain0):
    # W is the Nx1 vector; spiketrain0 is the 800 list, each value is the spiketrain of a signal neuron
    V = np.zeros(100)
    t = np.linspace(50, 150 - 1, 100)
    for neuron in range(0, 1600):
        for tspike in spiketrain0[neuron]:
            V = V + W[neuron] * Synaptic_activition(t - tspike)
    V_max = np.max(V)
    t_max = t[np.argmax(V)]
    return t_max, V_max, V

def spiketrain(spiketime,spikeindex,neuronnum=1600):
    spikelist = []
    for neuron in range(0, neuronnum):
        stime = spiketime[np.where(spikeindex == neuron)[0]]
        if len(stime) == 0:
            spikelist.append([])
        else:
            spikelist.append(stime)
    return spikelist
