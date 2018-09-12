import random
import numpy as np
import itertools

# The simple JMVAE environment

# HOWTO
#tmp = jmvae_gym('my_gym', 'x')
#tmp.sample_env()
#print(tmp.environment)
#tmp.step(0)
#print(tmp.environment)


class DataLoader:
    def __init__(self, filename):
        # Load and prepare the data
        data = np.load(filename)
        print(data.keys())
        self.x = data['x']
        self.w = data['w']
        self.v = data['v']

        self.z_x_mean = data['z_x_mean']
        self.z_w_mean = data['z_w_mean']
        self.z_v_mean = data['z_v_mean']
        self.z_xw_mean = data['z_xw_mean']
        self.z_xv_mean = data['z_xv_mean']
        self.z_wv_mean = data['z_wv_mean']
        self.z_xwv_mean = data['z_xwv_mean']

        self.z_w_var = data['z_w_var'] # are actually elbos
        self.z_x_var = data['z_x_var'] # are actually elbos
        self.z_v_var = data['z_v_var'] # are actually elbos
        self.z_xw_var = data['z_xw_var'] # are actually elbos
        self.z_xv_var = data['z_xv_var'] # are actually elbos
        self.z_wv_var = data['z_wv_var'] # are actually elbos
        self.z_xwv_var = data['z_xwv_var'] # are actually elbos

        self.label = data['label']
        self.info = data['info']
        data.close()
        # Get some information out of the VAE
        self.D_z = self.z_x_mean.shape[1] # Latent space dimension ( of the VAE)
        self.num_sample = self.label.shape[0] # Number of samples
        print(self.info)

    # samples an index for a given PoI
    def sample_idx_from_label(self, poi):
        p_ = np.float64(np.int32(self.label) == np.int32(poi))
        p = p_ / np.sum(p_)
        return np.random.choice(np.arange(self.num_sample), p=p)

class JmvaeGym_v2(DataLoader):
    def __init__(self, name, modality, permutate=False, has_distance=False):
        super(JmvaeGym_v2, self).__init__('./xwvz_set.npz')  # Load the data
        self.name = name
        # Quantities
        self.statistics = 2  # mu and sigma
        self.poi = 4  # number of objects
        self.position = np.zeros(self.poi) # 1-hot vector of current position [0,0,1,0,0]
        self.has_distance = has_distance # consideration object distance
        self.area_dim = [2,2] # dimension (x,y) of obversable area in meter
        self.objects = np.zeros((self.poi, 2)) # position of all poi
        self.obj_distance = np.zeros(self.poi) # distance from current position to all poi
        self.D_z_full = self.D_z * self.statistics  # full latent space dimension of the VAE
        self.observation_space = self.D_z * self.statistics * self.poi # statistics for all objects
        self.action_space = (self.poi-1)+1 # +1 for NOP   # choose one of the objects ( the additional action for quitting the game is added later)
        self.modality_space = 3  # x, w, v
        self.mean_idx = np.arange(0, self.D_z_full, self.statistics)  # the first mean occurs at index 0
        self.var_idx = np.arange(1, self.D_z_full, self.statistics)  # the first variance occurs at index 1
        # Environment
        self.environment = np.zeros((self.poi, self.D_z * self.statistics), dtype = np.float32)
        self.uninformed_mean = np.float32(0.0)
        self.uninformed_var = np.float32(2.)
        self.permutate = permutate  # permutate the PoI arrangements
        if not (modality == 'x' or modality == 'w' or modality == 'v'):
            raise Exception('modality needs to be x, w or v')
        else:
            self.modality = modality
        self.xwv_seen = np.zeros((self.action_space, self.modality_space), dtype = np.bool)
        self.sample_permutations = ["0123","0132", "0213", "0231", "0312", "0321",
                                    "1023", "1032", "1203", "1230", "1302", "1320",
                                    "2013", "2031", "2103", "2130", "2301", "2310",
                                    "3012", "3021", "3102", "3120", "3201", "3210"]  # all permutations of PoI arrangements
        self.current_permutation = self.sample_permutations[0]
        self.current_permutation_idx = np.zeros((self.poi,), dtype = np.int)
        # step information
        self.done = False

    def _sample_objects(self):
        self.objects = np.random.rand(self.poi, 2) * self.area_dim # sample random objects in area_dim, 2 for dim x,y
        self._update_distances()

    def _update_distances(self):
        if np.sum(self.position) == 0: # unknown pose
            pose = np.random.rand(1, 2) * self.area_dim # boostrap: sample robot to random position
        else:
            pose = self.objects[self._get_position()]
        self.obj_distance = np.array([np.linalg.norm(pose - self.objects[x]) for x in range(self.poi)])  # calc distance from current position to each other poi
        self.obj_distance /= (np.sqrt(np.power(self.area_dim[0], 2) + np.power(self.area_dim[1], 2)) * np.sqrt(2)) # normalize

    def _update_position(self, position):
        # position = -1 == unknown pose
        self.position = np.zeros(self.poi)  # reset current position
        if position >= 0: # pose known
            self.position[position] = 1  # set new position
        self._update_distances()

    def _get_position(self):
        # return current position
        # self.position: 1-hot vector of current position [0,0,1,0,0]
        return np.argmax(self.position)

    def _permutation2id(self):
        # Copies the string to the id lookup
        for idx in np.arange(self.poi):
            self.current_permutation_idx[idx] = int(self.current_permutation[idx])

    def _sample_permutation(self):
        # samples the permutation
        self.current_permutation = np.random.choice(self.sample_permutations)
        self._permutation2id()

    def _sample_seen(self):
        # return a vector of already seen PoIs
        return np.random.rand(self.action_space, self.modality_space) > .5

    def _encode(self, poi, modalities):
        environment = np.zeros((self.D_z * self.statistics), dtype = np.float32)  # (mu_1, sigma_1), ..., (mu_Dz, sigma_Dz)
        sample_idx = self.sample_idx_from_label(poi)
        if modalities == 'x':
            environment[0], environment[1] = self.z_x_mean[sample_idx, 0], self.z_x_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_x_mean[sample_idx, 1], self.z_x_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'w':
            environment[0], environment[1] = self.z_w_mean[sample_idx, 0], self.z_w_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_w_mean[sample_idx, 1], self.z_w_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'v':
            environment[0], environment[1] = self.z_v_mean[sample_idx, 0], self.z_v_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_v_mean[sample_idx, 1], self.z_v_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'xw':
            environment[0], environment[1] = self.z_xw_mean[sample_idx, 0], self.z_xw_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_xw_mean[sample_idx, 1], self.z_xw_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'xv':
            environment[0], environment[1] = self.z_xv_mean[sample_idx, 0], self.z_xv_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_xv_mean[sample_idx, 1], self.z_xv_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'wv':
            environment[0], environment[1] = self.z_wv_mean[sample_idx, 0], self.z_wv_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_wv_mean[sample_idx, 1], self.z_wv_var[sample_idx] # are actually elbos # HACK
        elif modalities == 'xwv':
            environment[0], environment[1] = self.z_xwv_mean[sample_idx, 0], self.z_xwv_var[sample_idx] # are actually elbos # HACK
            environment[2], environment[3] = self.z_xwv_mean[sample_idx, 1], self.z_xwv_var[sample_idx] # are actually elbos # HACK
        return environment

    def _get_lookup_poi(self, idx_poi):
        # provides the lookup for the actual PoI
        if not self.permutate:
            return idx_poi
        else:
            return self.current_permutation_idx[idx_poi]

    def _get_reverse_lookup_poi(self, idx_poi):
        # provides the lookup for the reversed PoI
        if not self.permutate:
            return idx_poi
        else:
            return np.argmax(self.current_permutation_idx == idx_poi)

    def get_state(self):
        return np.reshape(self.environment, [1, self.observation_space])


    def sample_env(self):
        self.done = False
        if self.has_distance:
            self._sample_objects() # sample positions of poi
            self._update_position(np.random.randint(self.poi)) # sample current position
        # sample the already seen PoIs
        self.xwv_seen = self._sample_seen()
        if self.modality == 'x':
            while np.all(self.xwv_seen[:,0]):
                self.xwv_seen = self._sample_seen()
        if self.modality == 'w':
            while np.all(self.xwv_seen[:,1]):
                self.xwv_seen = self._sample_seen()
        if self.modality == 'v':
            while np.all(self.xwv_seen[:,2]):
                self.xwv_seen = self._sample_seen()
        # Sample the permutation (usage TBD)
        self._sample_permutation()
        # Do the encoding wrt. the permutation
        for idx_poi in range(self.action_space):
            current_poi = self._get_lookup_poi(idx_poi)
            current_xwv_seen = self.xwv_seen[current_poi, :]
            if np.all(current_xwv_seen):  # sample x,w,v
                self.environment[current_poi, :] = self._encode(current_poi, 'xwv')
            elif current_xwv_seen[0] and current_xwv_seen[1]:  # sample x,w
                self.environment[current_poi, :] = self._encode(current_poi, 'xw')
            elif current_xwv_seen[0] and current_xwv_seen[2]:  # sample x,v
                self.environment[current_poi, :] = self._encode(current_poi, 'xv')
            elif current_xwv_seen[1] and current_xwv_seen[2]:  # sample w,v
                self.environment[current_poi, :] = self._encode(current_poi, 'wv')
            elif current_xwv_seen[0]:  # sample x
                self.environment[current_poi, :] = self._encode(current_poi, 'x')
            elif current_xwv_seen[1]:  # sample w
                self.environment[current_poi, :] = self._encode(current_poi, 'w')
            elif current_xwv_seen[2]:  # sample v
                self.environment[current_poi, :] = self._encode(current_poi, 'v')
            else:  # nothing seen yet
                self.environment[current_poi, 0] = self.uninformed_mean  # mu_1
                self.environment[current_poi, 1] = self.uninformed_var  # sigma_1
                self.environment[current_poi, 2] = self.uninformed_mean  # mu_2
                self.environment[current_poi, 3] = self.uninformed_var  # sigma_2

    def step(self, action):
        if action == self.action_space: # NOP terminating condition
            if self.modality == 'x' and (not self.xwv_seen[self._get_reverse_lookup_poi(0)][0] or not self.xwv_seen[self._get_reverse_lookup_poi(2)][0]):
                return self.get_state(), -1, True, {}
            elif self.modality == 'w' and (not self.xwv_seen[self._get_reverse_lookup_poi(1)][1] or not self.xwv_seen[self._get_reverse_lookup_poi(2)][1]):
                return self.get_state(), -1, True, {}
            elif self.modality == 'v' and (not self.xwv_seen[self._get_reverse_lookup_poi(2)][1] or not self.xwv_seen[self._get_reverse_lookup_poi(3)][1]):
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, True, {} # TODO: maybe return 1
        if action > self.action_space and action < 0:
            raise Exception('action needs to be within the action space')
        next_environment = np.copy(self.environment)  # local copy
        current_next_environment = np.copy(next_environment[action, :])
        current_xwv_seen = np.copy(self.xwv_seen[action, :])
        # Choose the proper change in the env. for the choosen action
        if self.modality == 'x':
            if np.all(current_xwv_seen[1:] == [0,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'x')
            elif np.all(current_xwv_seen[1:] == [1,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xw')
            elif np.all(current_xwv_seen[1:] == [0,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xv')
            elif np.all(current_xwv_seen[1:] == [1,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xwv')
            else:
                raise Exception('set x went wrong')
            current_xwv_seen[0] = True
        elif self.modality == 'w':
            if np.all(current_xwv_seen[[0,2]] == [0,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'w')
            elif np.all(current_xwv_seen[[0,2]] == [1,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xw')
            elif np.all(current_xwv_seen[[0,2]] == [0,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'wv')
            elif np.all(current_xwv_seen[[0,2]] == [1,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xwv')
            else:
                raise Exception('set w went wrong')
            current_xwv_seen[1] = True
        elif self.modality == 'v':
            if np.all(current_xwv_seen[:2] == [0,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'v')
            elif np.all(current_xwv_seen[:2] == [1,0]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xv')
            elif np.all(current_xwv_seen[:2] == [0,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'wv')
            elif np.all(current_xwv_seen[:2] == [1,1]):
                current_next_environment = self._encode(self._get_reverse_lookup_poi(action), 'xwv')
            else:
                raise Exception('set v went wrong')
            current_xwv_seen[2] = True
        else:
            raise Exception('I do not have this modality!')
        # assign the new envronment
        next_environment[action, :] = np.copy(current_next_environment)
        self.xwv_seen[action, :] = np.copy(current_xwv_seen)
        # reward
        elbo_old = np.mean(self.environment[action, self.var_idx])
        elbo_new = np.mean(next_environment[action, self.var_idx])
        #print("sigma_old:", sigma_old, "sigma_new:", sigma_new)
        #information_old = 1 / sigma_old if sigma_old != 0 else 0
        #information_new = 1 / sigma_new
        information_tmp = elbo_old - elbo_new
        #print("information_old:", information_old, "information_new:", information_new, "information_tmp:", information_tmp)
        if self.has_distance:
            dist = self.obj_distance[self._get_position()] # get distance from current position to action poi
        if action == self._get_position() and self.has_distance: # this should be a NOP and gets punished
                reward = -0.5
        else:
            if information_tmp < 0.1: #0.01 # we just have to shift the expected average reward a little bit
                reward = -1 #-np.abs(information_tmp) #if self.has_distance else -1.
            else:
                reward = information_tmp + (1-dist) if self.has_distance else information_tmp
        # done?
        if np.all(self.xwv_seen[:, 0]) and self.modality == 'x' or np.all(self.xwv_seen[:, 1]) and self.modality == 'w' or np.all(self.xwv_seen[:, 2]) and self.modality == 'v':
            self.done = True
        self.environment = next_environment
        self._update_position(action) # update position
        return self.get_state(), reward, self.done, {}