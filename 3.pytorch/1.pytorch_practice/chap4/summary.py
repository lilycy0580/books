
if __name__ == '__main__':
    """
    Chap4.神经网络工具箱nn
        1.nn.Module
        2.常用的神经网络层
            图像相关层
            激活函数
            构建神经网络
            循环神经网络
            损失函数
        3.nn.functional
            nn.functional与nn.Module的区别
            采样函数
        4.初始化策略
        5.优化器
        6.nn.Module深入分析
        7.搭建ResNet
    """
    """
    1.nn.Module
        1.torch.nn模块是构建在autograd之上的神经网络模块,专门为深度学习设计的模块,核心数据结构为Module
          torch.nn既可以表示为神经网络中的某个层layer,也可表示包含很多层的神经网络,常用做法就是继承nn.Module,然后编写自己的网络层
          
        2.nn.Module构建全连接层
            1.自定义层Linear需继承nn.Module
            2.构造函数__init__()中需自行定义可学习参数,并封装成nn.Parameter Parameter是特殊的Tensor,默认需要求导requeires_grad=True
            3.forward()实现前向传播
            4.反向传播无需手动编写,nn.Module根据autograd自动实现反向传播
            5.nn.Module中可学习参数通过named_parameters()或parameters()返回一个迭代器
            
        3.nn.Module实现多层感知机
            1.构造函数__init__()中,自定义Linear()层作为当前module对象的一个子module,子module的可学习参数,也会成为当前module的可学习参数
            2.forward()中可加上各层之间的处理函数(如激活函数,数据处理函数等),并定义层与层之间的关系
            3.module中Parameter的全局命名规范:
                对Parameter直接命名param_name 
                    self.param_name = nn.Parameter(t.randn(3,4))
                子moudle中的Parameter,添加当前module名称 
                    self.sub_module = SubModel()
                    self.sub_module.param_name
            4.Pytorch实现神经网络中绝大多数的网络层,这些层继承nn.Module,均封装可学习参数Parameter,并实现forward(),绝大多layer专门针对GPU进行cuDNN优化
            5.自定义layer对输入形状有规定:
                输入是一个batch数据,而非单个数据 当仅有一个数据时,需tensor.unsqueeze(0)或tensor[None]将数据伪装成batch_size=1的batch
            6.阅读源码文档nn.layer重点关键几点:
                构造函数的参数
                属性,可学习参数和包含的子module
                输入,输出的形状
    """
    """
    2.常用的神经网络层
        1.图像相关层
            图像相关层包含卷积层(Conv),池化层(Pool)等 这些层分为一维(1D),二维(2D),三维(3D)
                池化方式分为平均池化(AvgPool)  最大值池化(MaxPool) 自适应平均池化(AdaptiveAvgPool) 
                卷积层有前向卷积(Conv)和反卷积(TransposeConv)
            卷积神经网络的本质是卷积层,池化层,激活层以及其他层的叠加 
                
            1.卷积层
                torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)    
                    in_channels  输入图像的维度
                    out_channels 经过卷积操作后输出的维度
                    kernel_size  卷积核大小
                    stride       每次卷积操作移动的步长
                    padding      卷积操作在边界是否有填充,默认为0
                    bias         是否有偏置,默认为True
                    
                    输入形状 (N,C,H,W), 输出形状 (N,C,H,W)
                         H = { H+2*padding[0]-kernel_size[0] }/stride[0] + 1
                         W = { W+2*padding[1]-kernel_size[1] }/stride[1] + 1
                    
            2.池化层
                池化层可视为特殊的卷积层,主要用于下采样,池化层在保留主要特征的同时降低参数量,防止过拟合
                池化层无可学习的参数,weight固定,最长的是最大池化核平均池化
                
            3.其他层
                Linear 全连接层
                BatchNorm 批标准化层,分为1D,2D和3D
                Dropout 防止过拟合,分为1D,2D和3D
            
        2.激活函数
            线性模型不能解决所有问题,使用激活函数增加增加非线性因素,提高模型的表达能力
            ReLU(x) = max(0,x)
            tanh(x)
            sigmoid(x)
                
        3.构建神经网络
            前馈神经网络 
                每一层的输出都是下一层的输入
                Sequential 和 ModuleList
                
        4.循环神经网络
            自动处理整个序列的循环计算:
                RNN LSTM GRU
            只处理序列中的一个时间步(一个step):
                RNNCell LSTMCell GRUCell      
            
        5.损失函数
            略
    """

    """
    3.nn.functional
        1.torch.nn.functional 
            torch.nn中大多数layer,在nn.functional中都有对应的函数
            nn.functional vs nn.Module
                区别:
                    使用nn.functional实现的layer是个函数
                    使用nn.Module实现的layer是个类
                使用:    
                    当模型具有可学习的参数,使用nn.Module,否则两者均可
                    激活函数和池化层 nn.functional
                    卷积层和全连接层 nn.Module
                    dropout       nn.Dropout
        2.采样函数 
            略
    """
    """
    4.初始化策略
        深度学习中,参数的初始化十分重要 nn.init
            Pytorch中,nn,Module中的模块参数都采取较为合理的初始化策略,用户无需再设计
            用户也可使用自定义初始化策略
                使用Parameter时,自定义初始化很重要,torch.Tensor()返回的是内存中的随机数,可能在模型训练时造成溢出或梯度消失
    """
    """
    5.优化器
        Pytorch将深度学习中常见的优化方法全部封装到torch.optim中,所有优化方法全部继承torch.optim.Optimizer,并实现自己的优化步骤
            eg:SGD 
        调整学习率的两种方式:
            修改optimizer.param_groups种对应的学习率
            新建一个优化器  optimizer十分轻量级,构建新的开销小
                         但是新建优化器会重新初始化动量等状态信息,对于使用动量的优化器如adam来说,会导致损失函数在收敛过程种出现震荡等情况
    """
    """
    6.nn.Module
        1.nn.Module基类的构造函数
            def __init__(self):
                self._parameters = OrderedDict()   
                    保存用户直接设置的Parameter 
                        self.param1 = nn.Parameter(torch.randn(3,4)) key = param1, value = Parameter的item 
                        self.submodule = nn.Linear(3,4)中的parameter不会被保存到此字典种
                
                self._modules = OrderedDict()
                    子modules,self.submodule = nn.Linear(3,4)指定的子module会被保存于此
                
                self._buffers = OrderedDict()
                    缓存,BatchNorm使用动量机制,每次前向传播时都需要用到上一次前向传播的结果
                
                self._forward_hooks = OrderedDict()
                self._backward_hooks = OrderedDict()
                    钩子计数,用来提取中间变量
            
                self.training = True
                    BatchNorm层与Dropout层在训练阶段和测试阶段采取的策略不同,通过training属性决定前向传播策略
            
            注意:
                1.通过self.key获取_parameters,_modules,_buffers中的键值对 
                2.nn.Module在实际使用中可能层层嵌套,一个module可能包含若干个子module,每一个module也可能包含更多的子module
                    children()  查看所有直接子module
                    modules()   查看所有子module
                    named_children()
                    named_modules()
                3.Dropout层,BatchNorm层等在训练阶段和测试阶段时采取的策略不同,通过设置trainning切换不同的前向传播策略
                4.设置training,将子module分为train模式和eval模式
                    model.train() 将当前module及其子module中所有的training=True
                    model.eval()  将所有training=False
                5.查看中间层变量的梯度,使用钩子函数
                    register_forward_hook(forward)
                    register_backward_hook(backward) 
                    
                    在前向传播或反向传播时注册钩子函数,每次前向传播结束后会执行钩子函数
                        hook(module,input, output) -> None
                        hook(module, grad_input, grad_output) -> Tensor or None    
                    
                    钩子函数不应该修改模型的输入和输出,使用后应该及时删除,避免每次都运行钩子函数增加负载
                    钩子函数主要用于获取中间结果
                        eg:获取网络中间某一层的输出或某一层的梯度
                
                6.两个魔法方法
                    __getattr__ 
                    __setattr__
                    
                    Python中两个buildin方法:
                        getattr(obj,'attr1')        obj.attr1
                        setattr(obj,'name',value)   obj.name = value
                        
                        如果getattr()无法找到所需的属性,Python会调用obj.__getattr__('attr') 
                        若对象没有实现__getattr__(),或者遇到__getattr__()无法处理的情况,则程序抛异常
                        
                        若对象实现__setttr__(),则setattr会直接调用obj.__setattr__('name',value),否则调用buildin方法
                        
                    总结:
                        result = obj.name,调用buildin方法getattr(obj,'attr'),若该属性找不到,则调用obj.__getattr__('name')
                        obj.name = value,调用buildin方法setattr(obj,'name',value)
                                         若obj对象实现了__setattr方法,则setattr直接调用obj.__setattr__('name',value) 
                    
                    nn.Module实现自定义的__setattr__方法,
                        当执行module.name=value时,会在__setattr__中判断value是否为Parameter或nn.Module对象
                            若是,则将这些对象加入_parameters和_modules两个字典中
                            若是其他类型的对象,如list,dict等,则调用默认操作,将对象保存在__dict__中
                    
                    _modules和_parameters中的item没有被保存在__dict__中,默认的getattr()无法获取它,所以nn.Module实现了自定义的__getattr__()
                    若默认的getattr()无法处理,则调用自定义的__getattr__(),尝试从_modules,_parameters和_buffers这三个字典中获取
                    
                7.保存模型
                    Pytorch保存模型十分简单,所有module对象都具有state_dict()函数,返回当前module的所有状态数据,将这些状态数据保存后,下次使用模型时即可利用model.load_staate_dict()函数将状态加载进来
                    优化器也有类型的机制
                
                8.GPU上运行模型
                    model.cuda()  将模型的所有参数都转存到GPU上
                    input.cuda()  将输入数据放置到GPU上
                    
                    在多块GPU上进行并行计算
                        1.nn.parallel.data_parallel(model,input,device_ids=Node,output_device=Node,dim=0,module_kwargs=None)
                            利用多块GPU进行并行计算并获取结果
                        
                        2.class torch.nn.DataParallel(module,device_ids=Node,output_device=Node,dim=0)
                            返回一个新的module,能够自动在多个GPU上进行并行加速
                            DataParallel是将一个batch的数据均分为多份,分别送到对应的GPU上进行计算,然后将各块GPU上得到的梯度进行累加,与module相关的所有数据也会以浅复制的方式复制多份
                            
                        device_ids    指定在那些GPU上进行优化     
                        output_device 指定输出到哪块GPU上        
    
                        
                        
                        
    
    
    
    
    
    
    
    
    
    """

    """
    7.ResNet
        ResNet的结构解决训练极深网络时的梯度消失问题
        ResNet变种,ResNet34 
    """