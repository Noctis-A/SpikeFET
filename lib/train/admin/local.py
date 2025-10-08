class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/data/yjj/SpikeFET'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/data/yjj/SpikeFET/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/data/yjj/SpikeFET/pretrained_networks'
        self.coesot_val_dir = '/mnt/data/yjj/SpikeFET/data/COESOT/test'
        self.coesot_dir = '/mnt/data/yjj/SpikeFET/data/COESOT/train'
        self.fe108_dir = '/home/work/yjj/FE108/train'
        self.fe108_val_dir = '/home/work/yjj/FE108/test'
        self.visevent_dir = '/home/work/yjj/VisEvent/train'
        self.visevent_val_dir = '/home/work/yjj/VisEvent/test'
