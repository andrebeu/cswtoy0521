import numpy as np

NSTATES = 10


class SchemaTabularBayes():
    """ CRP prior
    tabluar predictive distirbution
    """
    def __init__(self,concentration,stickiness,sparsity):
        self.Tmat = np.zeros([NSTATES,NSTATES])
        self.alfa = concentration
        self.beta = stickiness
        self.lmbda = sparsity
        self.ntimes_sampled = 0
        self.active = 0 # 1 if active on previous step

    def get_prior(self):
        if self.ntimes_sampled == 0:
            return self.alfa
        crp = self.ntimes_sampled + self.beta*self.active
        return crp

    def get_like(self,xtm1,xt):
        num = self.lmbda + self.Tmat[xtm1,xt]
        den = (NSTATES*self.lmbda) + self.Tmat[xtm1,:].sum()
        like = num/den
        return like

    def update(self,xtm1,xt):
        self.Tmat[xtm1,xt]+=1
        return None

    def activate(self,flip):
        if flip:
            self.active = 1
            self.ntimes_sampled += 1
        else:
            self.active = 0
        return None



class SEM():
    def __init__(self,schargs):
        self.SchClass = SchemaTabularBayes
        self.schargs = schargs
        self.init_schlib()

    def init_schlib(self):
        """ 
        initialize with two schemas
        one active one inactive
        """
        sch0 = self.SchClass(**self.schargs)
        sch1 = self.SchClass(**self.schargs)
        self.schlib = [sch0,sch1]

    def calc_posteriors(self,xtm1,xt):
        """ loop over schema library
        """
        priors = [sch.get_prior() for sch in self.schlib]
        likes = [sch.get_like(xtm1,xt) for sch in self.schlib]
        posteriors = [p*l for p,l in zip(priors,likes)]
        return posteriors

    def select_sch(self,xtm1,xt):
        """ xt and xtm1 are ints
        """
        posteriors = self.calc_posteriors(xtm1,xt)
        active_k = np.argmax(posteriors)
        if active_k == len(self.schlib)-1:
            self.schlib.append(self.SchClass(**self.schargs))
        return active_k

    def predict(self,xtm1):
        """ 
        """
        pr_xt_z = np.array([
            self.calc_posteriors(xtm1,x) for x in range(NSTATES)
            ]) # probability of each next state under each schema
        pr_xtp1 = np.sum(pr_xt_z,axis=1) # sum over schemas
        return pr_xtp1

    def run_exp(self,exp):
        """ exp is L of trialL
        trialL is L of obs (ints) 
        """
        ## recording
        Mt0ij = np.zeros([len(exp),NSTATES,NSTATES])
        data = {
            'zt':-np.ones([len(exp),len(exp[0])]),
            'xth':-np.ones([len(exp),len(exp[0]),NSTATES])
        }
        ## 
        schtm = self.schlib[0] # sch0 is active to start
        for tridx,trialL in enumerate(exp):
            for tstep,(xtm,xt) in enumerate(zip(trialL[:-1],trialL[1:])):
                xth = self.predict(xtm)
                # update infered active schema
                zt = self.select_sch(xtm,xt)
                scht = self.schlib[zt]
                # update transition matrix
                scht.update(xtm,xt)
                # update schema history
                schtm.activate(0)
                scht.activate(1)
                schtm = scht
                data['xth'][tridx][tstep] = xth
                data['zt'][tridx][tstep] = zt               
        return data






class SchemaDeltaLearner():
    """ no prior
    tabluar learner 
    delta rule updates randomly initialized distribution 
    """
    def __init__(self,init_lr=0.3,lr_decay_rate=0.1):
        self.nstates = STSPACE_SIZE
        # paramS
        self.init_lr = init_lr  # fit
        self.lr_decay_rate = lr_decay_rate # fit; larger faster decay 
        # init objects
        self.Tmat = self._init_transition_matrix()
        self.nupdates = 1

    def _init_transition_matrix(self):
        # T[s0,s1] = pr(s1|s0)
        T = np.random.random((self.nstates,self.nstates))
        T = np.transpose(T/T.sum(axis=0)) # rows sum to one
        return T

    def calc_error_obs(self,st0,st1):
        """ delta err vec is len Nstates
        O(st0) - pr(st1), where pr(st1) 
        is next state prediciton (softmax)
        """
        obs = np.zeros(self.nstates)
        obs[st1] = 1
        delta_err_vec = obs-self.Tmat[st0]
        return delta_err_vec

    def calc_error_on_path(self,path):
        """ returns {st0: delta_err_vec} for st0 in path
        """
        D = {}
        for st0,st1 in zip(path[:-1],path[1:]):
            D[st0] = self.calc_error_obs(st0,st1)
        return D

    def update_sch(self,path):
        lr=self.get_lr()
        errD = self.calc_error_on_path(path)
        for st0,errvec in errD.items():
            self.Tmat[st0,:] += lr*errvec
        self.nupdates += 1
        return None

    def get_lr(self):
        return self.init_lr*np.exp(-self.lr_decay_rate*self.nupdates)

    def eval(self):
        """ 
        eval schema response on all paths
        returns [npaths,nsteps] 
        where each entry is probability of correct response
        """
        task = Task()
        paths = [item for sublist in task.paths for item in sublist]
        acc_arr = []
        for path in paths:
            acc_arr.append(self.eval_path(path))
        return np.array(acc_arr)

    def eval_path(self,path):
        accL = []
        for s0,s1 in zip(path[:-1],path[1:]):
            accL.append(self.Tmat[s0,s1])
        return np.array(accL)

    def calc_pe(self,path):
        errD = self.calc_error_on_path(path)
        pe = np.sum([i**2 for i in list(errD.values())])
        return pe






class Task():
    """ 
    """

    def __init__(self):
        A1,A2,B1,B2 = self._init_paths_csw()
        self.paths = [[A1,A2],[B1,B2]]
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None


    def _init_paths_csw(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        begin = 0
        locA,locB = 1,2
        node11,node12 = 3,4
        node21,node22 = 5,6
        node31,node32 = 7,8
        end = 9
        A1 = np.array([begin,locA,
            node11,node21,node31,
            end
            ])
        A2 = np.array([begin,locA,
            node12,node22,node32,
            end
            ])
        B1 = np.array([begin,locB,
            node11,node22,node31,
            end
            ])
        B2 = np.array([begin,locB,
            node12,node21,node32,
            end
            ])
        return A1,A2,B1,B2

    def _init_paths_toy(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        locA,locB = 0,1
        node11,node12 = 2,3
        node21,node22 = 4,5
        A1 = np.array([locA,
            node11,node21
            ])
        A2 = np.array([locA,
            node12,node22
            ])
        B1 = np.array([locB,
            node11,node22
            ])
        B2 = np.array([locB,
            node12,node21
            ])
        return A1,A2,B1,B2


    def get_curriculum(self,condition,n_train,n_test):
        """ 
        order of events
        NB blocked: ntrain needs to be divisible by 4
        """
        curriculum = []   
        if condition == 'blocked':
            assert n_train%4==0
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4 )
        elif condition == 'early':
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 4)
        elif condition == 'middle':
            curriculum =  \
                [0, 1] * (n_train // 8) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 8)
        elif condition == 'late':
            curriculum =  \
                [0, 1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4)
        elif condition == 'interleaved':
            curriculum = [0, 1] * (n_train // 2)
        elif condition == 'single': ## DEBUG
            curriculum =  \
                [0] * (n_train) 
        else:
            print('condition not properly specified')
            assert False
        # 
        curriculum += [int(np.random.rand() < 0.5) for _ in range(n_test)]
        return np.array(curriculum)


    def generate_experiment(self,condition,n_train,n_test):
        """ 
        exp: arr [ntrials,tsteps]
        curr: arr [ntrials]
        """
        # get curriculum
        n_trials = n_train+n_test
        curr = self.get_curriculum(condition,n_train,n_test)
        # generate trials
        exp = -np.ones([n_trials,self.tsteps],dtype=int)
        for trial_idx in range(n_train+n_test):
            # select A1,A2,B1,B2
            event_type = curr[trial_idx]
            path_type = np.random.randint(2)
            path_int = self.paths[event_type][path_type]
            # embed
            exp[trial_idx] = path_int
        return exp,curr