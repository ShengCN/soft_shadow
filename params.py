import argparse

class params():
    """ Singleton class for doing experiments """
    
    class __params():
        def __init__(self):
            self.norm = 'group_norm'
            self.activation = 'relu'
            self.bilinear = True
            self.prelu = False
            
        def set_params(self, options):
            self.options = options
            self.norm = options.norm
            self.bilinear = options.bilinear
            
        def __str__(self):
            return 'norm: {} bilinear: {} activation: {}'.format(self.norm, self.bilinear, self.activation)
        
    # private static variable
    param_instance = None
    
    def __init__(self):
        if not params.param_instance:
            params.param_instance = params.__params()
    
    def get_params(self):
        return params.param_instance
    
    def set_params(self, options):
        params.param_instance.set_params(options)

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batch_size', type=int, default=28, help='input batch size during training')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate, default=0.005')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum for SGD, default=0.9')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--weight_file',type=str,  help='weight file')
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPU training')
    parser.add_argument('--timers', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--use_schedule', action='store_true',help='use automatic schedule')
    parser.add_argument('--exp_name', type=str, default='l1 loss',help='experiment name')
    parser.add_argument('--new_exp', action='store_true', help='experiment 2')
    parser.add_argument('--bilinear', action='store_true', help='use bilinear in up-stream')
    parser.add_argument('--norm', type=str, default='batch_norm', help='use group norm')
    
    # parser.add_argument('--cpu', action='store_true', help='Force training on CPU')
    arguments = parser.parse_args()
    parameter = params()
    parameter.set_params(arguments)
    
    return arguments