import torch


# ==================================== WEIGHT INITIALIZER ===========================================
class WeightInitializer:
    '''
    Utiility class for initializing the weights of a network.

    Usage example:
        weightInit = WeightInitializer()
        weightInit.init_weights(model, 'xavier_normal_', {'gain':0.02})

    '''

    def __init__(self, initType = None, kwargs = { }):
        self.kwargs = kwargs
        self.weightInit = None

        if initType is not None:
            if not hasattr(torch.nn, initType):
                raise NotImplementedError('Init method [%s] does not exist in torch.nn' % initType)
            self.weightInit = getattr(torch.nn.init, initType)

    # ===============================================  INIT WEIGHTS =================================
    def init_weights(self, model, weightInit = None, kwargs = { }):
        '''
        Function called for initializeing the weights of a model
        :param model: pytorch model
        :param weightInit: init type (must be in torch.nn.init.*)
        :param kwargs: kwargs to be passed to the initialization function
        :return:
        '''

        if weightInit is not None:
            if not hasattr(torch.nn.init, weightInit):
                raise NotImplementedError('Init method %s not in torch.nn' % weightInit)
            self.weightInit = getattr(torch.nn.init, weightInit)

        self.kwargs = kwargs if kwargs != { } else self.kwargs

        model.apply(self._init_module)

    # =============================================== INIT MODULES ====================================================
    def _init_module(self, module):
        '''
        Internal function which is applied to every module in a network

        :param module: model to be applied to
        '''

        className = module.__class__.__name__

        # init conv and linear layers
        if hasattr(module, 'weight') and (className.find('Conv') != -1 or className.find('Linear') != -1):
            self.weightInit(module.weight.data, **self.kwargs)
            # init biases
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)

        # init batch norm weightd
        elif className.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(module.bias.data, 0.0)

    # =============================================== INIT NET =====================================================
    def parallel_net(self, net, gpus = []):

        assert (torch.cuda.is_available()), 'Cuda is not available'
        assert len(gpus) > 0, 'GPU id not specified'
        net = torch.nn.DataParallel(net, gpus)  # multi-GPUs

        return net
