class hyperparameter():
    def __init__(self):
        self.model_name = "KIBA"
        self.hid_dim = 64
        self.n_layers = 3
        self.n_heads = 8
        self.pf_dim = 256 
        self.dropout = 0.1
        self.batch = 64
        self.d_ff = 512
        self.lr = 1e-4
        self.weight_decay = 1e-6
        self.iteration = 100 
        self.n_folds = 5
        self.seed = 2022
        self.save_name = "test"
        self.MAX_PROTEIN_LEN = 620
        self.MAX_DRUG_LEN = 100
        self.atom_dim = 34
        self.pro_em = 9050
        self.smi_em = 82