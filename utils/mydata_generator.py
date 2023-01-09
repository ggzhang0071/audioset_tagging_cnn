class BalancedTrainSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
         Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedTrainSampler, self).__init__(indexes_hdf5_path,batch_size,black_list_csv,random_seed)
        self.samples_num_per_class=np.sum(self.target,axis=0)
        logging.info('samples_num_per_class:{}'.format(self.samples_num_per_class))

        self.indexes_per_class=[]
        for i in range(self.classes_num):
            self.indexes_per_class.append(np.where(self.target[:,i]==1)[0])
        
        # 
                

