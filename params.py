import argparse

class params():
    """ Singleton class for doing experiments """
    
    class __params():
        def __init__(self):
            self.norm = 'batch_norm'
            self.activation = 'relu'
            self.bilinear = True
            self.prelu = False
            self.double_conv = False
            self.ibl_num = 3
            self.weight_decay = 5e-4
            self.scale_ibl = False
            self.small_ds = False
            self.new_ibl = False
            self.multi_gpu = False
            self.log = True
            self.coordconv = False

        def set_params(self, options):
            self.options = options
            self.norm = options.norm
            self.bilinear = options.bilinear
            self.prelu = options.prelu
            self.double_conv = options.double_conv
            self.ibl_num = options.ibl_num
            self.weight_decay = options.weight_decay
            self.scale_ibl = options.scale_ibl
            self.small_ds = options.small_ds
            self.new_ibl = options.new_ibl
            self.multi_gpu = options.multi_gpu
            self.log = options.log
            self.coordconv = options.coordconv

        def __str__(self):
            return 'norm: {} bilinear: {} activation: {} prelu: {} ibl: {} weight decay: {} scale_ibl: {} small ds: {}'.format(self.norm, 
                                                                                                                               self.bilinear, 
                                                                                                                               self.activation,
                                                                                                                               self.prelu,
                                                                                                                               self.ibl_num,
                                                                                                                               self.weight_decay,
                                                                                                                               self.scale_ibl,
                                                                                                                               self.small_ds)

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
    parser.add_argument('--relearn', action='store_true', help='forget previous best validation loss')
    parser.add_argument('--weight_file',type=str,  help='weight file')
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPU training')
    parser.add_argument('--scale_ibl', action='store_true', help='scale the (ibl, shadow) pair')
    parser.add_argument('--timers', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--use_schedule', action='store_true',help='use automatic schedule')
    parser.add_argument('--patience', type=int, default=2, help='use automatic schedule')
    parser.add_argument('--exp_name', type=str, default='l1 loss',help='experiment name')
    parser.add_argument('--new_exp', action='store_true', help='experiment 2')
    parser.add_argument('--bilinear', action='store_true', help='use bilinear in up-stream')
    parser.add_argument('--norm', type=str, default='batch_norm', help='use group norm')
    parser.add_argument('--prelu', action='store_true', help='use group norm')
    parser.add_argument('--double_conv', action='store_true', help='use group norm')
    parser.add_argument('--small_ds', action='store_true', help='small dataset')
    parser.add_argument('--new_ibl', action='store_true', help='use new ibl represetation') 
    parser.add_argument('--log', action='store_true', help='log information')
    parser.add_argument('--ibl_num', type=int, default=24, help='maximum ibl number during training')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay for model weight')
    parser.add_argument('--save', action='store_true', help='save batch results?')
    
    parser.add_argument('--coordconv', action='store_true', help='use coord convolution')

    # parser.add_argument('--cpu', action='store_true', help='Force training on CPU')
    arguments = parser.parse_args()
    parameter = params()
    parameter.set_params(arguments)
    
    return arguments