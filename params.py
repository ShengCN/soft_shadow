class params():
    """ Singleton class for doing experiments """
    class __params():
        def __init__(self):
            self.trans_conv_kernel_size = 2
            # self.up_layer= 'trans_conv'
            self.up_layer= 'bilinear'
        
        def set_params(self, options):
            self.options = options
            
    # private static variable
    param_instance = None
    
    def __init__(self):
        if not params.param_instance:
            params.param_instance = params.__params()
    
    def get_params(self):
        return params.param_instance.options
    
    def set_params(self, options):
        params.param_instance.set_params(options)