import paddle.fluid as fluid


def conv3x3(num_channels,num_filters,stride=1,groups=1,dilation=1):
    '''
    3x3 convolution with padding
    dilation = padding ensures the size 
    '''
    return fluid.dygraph.Conv2D(num_channels=num_channels,num_filters=num_filters,filter_size=3,stride=stride,
                                padding=dilation,groups=groups,dilation=dilation)
def conv1x1(num_channels,num_filters,stride=1):
    '''
    1x1 convolution
    changes the number of planes
    '''
    return fluid.dygraph.Conv2D(num_channels=num_channels,num_filters=num_filters,filter_size=1,stride=stride,bias_attr=False)

class Bottleneck(fluid.dygraph.Layer):
    # Bottleneck here places the stride for downsampling at 3x3 convolution
    # while the original implementation places the stride at the first 1x1 convolution
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None,
                groups=1, base_width =64,dilation=1,norm_layer=None):
        super().__init__()
        if norm_layer == None:
            norm_layer=fluid.dygraph.BatchNorm
        width = int(planes * (base_width/64.)) * groups
        self.conv1 = conv1x1(inplanes,width)
        self.bn1 =  norm_layer(width)
        self.conv2 = conv3x3(width,width,stride,groups,dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width,planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        ## self.relu = fluid.layers.relu  ??? question

    def forward(self,inputs):
        res = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = fluid.layers.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            res = self.downsample(inputs)
        x = fluid.layers.elementwise_add(x,res)
        x = fluid.layers.relu(x)
        return x

class BasicBlock(fluid.dygraph.Layer):

    expansion = 1 # This valuable can be ignored
    
    def __init__(self,inplanes,planes,stride=1,downsample=None,
                groups=1, base_width =64,dilation=1,norm_layer=None):
        super().__init__()
        if norm_layer == None:
            norm_layer = fluid.dygraph.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only support groups = 1 and base_width =64')
        if dilation > 1:
            raise NotImplementedError('BasicBlock only support dilation = 1')
        
        self.conv1 = conv3x3(inplanes,planes,stride=stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.downsample=downsample
        self.stride = stride
    def forward(self,inputs):
        res = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            res = self.downsample(inputs)
        x = fluid.layers.elementwise_add(x,res)
        x = fluid.layers.relu(x)
        return x

class ResNet(fluid.dygraph.Layer):
    def __init__(self,block,layers,num_classes = 100, zero_init_residual = False,
                groups=1,width_per_group=64,replace_stride_with_dilation=None,
                norm_layer=None):
        super(ResNet,self).__init__()
        if norm_layer == None:
            norm_layer = fluid.dygraph.BatchNorm
        self._norm_layer = norm_layer
        # initialize the inplanes syntax 
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation == None:
            replace_stride_with_dilation = [False,False,False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or'
                            ' a 3-element tuple, got{}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = fluid.dygraph.Conv2D(3,self.inplanes,filter_size=7,stride=2,padding=3,bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = fluid.dygraph.Pool2D(pool_size=3,pool_type='max',pool_stride=2,pool_padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])

        self.layer2 = self._make_layer(block,128,layers[1],stride=2,dilation = replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block,256,layers[2],stride=2,dilation = replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block,512,layers[3],stride=2,dilation = replace_stride_with_dilation[2])
        self.fc = fluid.dygraph.Linear(512*block.expansion,num_classes)
        
        # TODO: initialize the parameters(weights) and biaes


    def _make_layer(self,block,planes,blocks,stride=1,dilation = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilation:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fluid.dygraph.Sequential(
                conv1x1(self.inplanes,planes * block.expansion,stride),
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample,
                            self.groups,self.base_width,previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes,groups=self.groups,
                                base_width=self.base_width,dilation=self.dilation,
                                norm_layer=norm_layer))
        return fluid.dygraph.Sequential(*layers)
    
    def _forward_impl(self,inputs):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = fluid.layers.adaptive_pool2d(x,pool_size=(1,1),pool_type='avg')
        x = fluid.layers.flatten(x,1)
        x = self.fc(x)

        return x

    def forward(self,inputs):
        return self._forward_impl(inputs)

def _resnet(block,layers,pretrained,**kwargs):  # ignore some valuables
    model = ResNet(block,layers,**kwargs)
    if pretrained:
        pass
        # TODO load pretrained weights
    return model

def resnet18(pretrained=False,**kwargs):
    return _resnet(BasicBlock,[2,2,2,2],pretrained,**kwargs)

def resnet50(pretrained=False,**kwargs):
    return _resnet(Bottleneck,[3,4,6,3],pretrained,**kwargs)