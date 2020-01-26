class params():
    """ Singleton class for doing experiments """
    class __params():
        def __init__(self):
            self.trans_conv_kernel_size = 2
    
    # private static variable
    param_instance = None
    
    def __init__(self):
        if not params.param_instance:
            params.param_instance = params.__params()
        
    def get_trans_conv_kernel(self):
        return params.param_instance.trans_conv_kernel_size
    
    def set_trans_conv_kernel(self, kernel_size):
        params.param_instance.trans_conv_kernel_size = kernel_size