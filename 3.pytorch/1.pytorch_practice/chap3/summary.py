if __name__ == '__main__':
    """
    Chap3.Tensor 与 autograd
        1.Tensor基础:
            1.Tensor的基本操作
                创建Tensor
                Tensor的类型
                索引操作
                拼接操作
                高级索引
                逐元素操作
                归并操作
                比较操作
                其他操作
            2.命名张量
            3.Tensor与Numpy
            4.Tensor的基本结构
            5.N种改变Tensor形状的方法
                查看维度
                改变维度
                Tensor的转置  
                
        2.线性回归
            略
            
        3.autograd和计算图基础
            1.autograd用法,requires_grad与backward
            2.autograd原理,计算图
            3.扩展autograd,Function
            4.利用autograd实现线性回归  
    """

    """    
    Tensor可以是一个数(标量),一维数组(向量),二维数组(矩阵,黑白图像)或更高维数组(高阶数据,视频等)
    Pytorch的Tensor支持GPU加速  

    Tensor接口与Numpy接口类似,对Tensor的操作可分为以下两类:
        1.从接口角度
            torch.function      torch.save      torch.sum(a,b) 等价于 a.sum(b)
            tensor.function     tensor.view
        2.从存储角度
            不修改自身存储的数据操作    a.add(b),返回一个新的Tensor
            修改自身存储的数据操作      a.add_(b),结果存储在a中,并返回此结果  inplace方式   
    """

    """
    1.Tensor基础
        1.Tensor的基本操作
            1.创建Tensor
                1.常见的创建Tensor方法 11种 略 
                    11种方法均可在创建Tensor时指定数据类型dtype和存放设备device
                    torch.Tensor()的参数:
                        接收一个list,根据list创建Tensor
                        接收指定的形状创建Tensor
                        接收其他的Tensor
                2.torch.Tensor(*sizes)创建Tensor时不会立刻分配空间,只有在真正使用创建的Tensor时才分配空间,其他操作都是在创建完Tensor后立刻进行空间分配
                3.torch.Tensor() vs torch.tensor()
                    torch.Tensor()
                        Python类,默认是torch.FloatTensor()
                    torch.tensor()
                        Python函数,直接从data中进行数据复制,并根据其类型创建相应类型的Tensor 推荐
    
            2.Tensor的类型
                1.Tensor类型分为设备类型和数据类型 device与dtype 
                    device:CPU与GPU          tensor.device()
                    dtype:bool,int,float     tensor.dtype()
                2.Tensor默认数据类型为FloatTensor 
                    torch.set_default_tensor_type修改Tensor类型 Tensor类型每个元素占32bit,即4B
                3.HalfTensor
                    专门为GPU设计的,显存仅占FloatTensor的一半,缓解显存不足
                    表示的数值大小和精度有限,可能会出现数据溢出问题
                4.不同类型的Tensor之间相互转换
                    类型转换:
                        tensor.type()   tensor.float() tensor.long() tensor.half()
                    CPU与GPU转换:
                        tensor.cuda()与tensor.cpu() 
                        tensor.device()
                    创建同种类型的Tensor
                        tensor.*_like()  tensor.*_like(tensorA)生成与tensorA相同属性的新Tensor    
                        tensor.new_*     tensor.new_*(new_shape)生成一个形状不同但属性相同的Tensor
    
            3.索引操作
                1.使用索引操作获取指定位置的数据
                    大多数索引操作通过修改Tensor的stride等属性与原Tensor共享内存,即修改了其中一个Tensor,另一个Tensor也会改变
                2.Pytorch中的选择函数 
                    4个 略
                    
            4.拼接操作
                cat():多个Tensor在指定维度dim上进行拼接
                stack():多个Tensor沿一个新的维度进行拼接,即在新维度上进行拼接
    
            5.高级索引
                略
                
            6.逐元素操作
                常见操作函数 略
                
            7.归并操作
                常见归并函数 略  
                dim维度:
                    若输入形状(m,n,k),则存在以下三种情况:
                        dim=0,输出形状为(1,n,k)或(n,k) keepdim=True则会保留维度1 
                        dim=1,输出形状为(m,1,k)或(m,k)
                        dim=2,输出形状为(m,n,1)或(m,n)
                
            8.比较操作
                部分函数是逐元素比较,部分函数类似归并操作
                常见比较函数 略
                    eg:
                        torch.max(tensor)
                        torch.max(tensor,dim)
                        torch.max(tensor,tensor2)
                            
            9.其他操作
                torch.fft 傅里叶变换
                torch.linalg 线性代数模块
                torch.distributions 概率分布和采样模块 伯努利分布,柯西分布,正态分布,拉普拉斯分布
    """

    """
        2.命名张量
            略
    
        3.Tensor与Numpy
            1.Tensor与Numpy大多数共享内存,当CPUTensor不支持操作时,先将其转为Numpy数组,完成相应处理后再转回Tensor
            2.torch.Tensor()创建的张量默认dtype=float32,若Numpy类型与默认类型不一致,则数据仅被复制,不共享内存
            3.torch.tensor()仅进行数据复制,不会共享内存
                注意torch.Tensor(),torch.from_numpy(),torch.tensor()在内存共享的区别
    
        4.Tensor的基本结构
            1.Tensor结构:   
                头信息区 保存Tensor的形状,步长,数据类型等信息 shape,stride,type   图略
                存储区   存储数据
                注意:
                    Tensor的内存地址执行Tensor的头,不同Tensor的头信息一般不同,但可使用相同的存储区
            2.绝大多数操作不修改Tensor的存储区,而修改Tensor的头信息,节约内存+提升处理速度
              有些操作会导致Tensor不连续,tensor.contiguous()变为连续的数据,该方法复制数据到新内存,不与原数据共享存储区
            
        5.改变Tensor形状的N种方法
            1.查看维度
                tensor.shape
                tensor.size()   等价于tensor.shape
                tensor.dim()    等价于len(tensor.shape),查看Tensor的维度
                tensor.numel()  查看Tensor中元素的数量
            
            2.改变维度
                tensor.reshape() 将内存中不连续的Tensor变连续后在进行形状变化
                reshape vs view:
                    reshape,等价于tensor.contiguous().view() 将内存不连续的Tensor先复制,调用contiguous()变连续后在改变形状,可避免报错  推荐使用
                    view,仅处理内存连续的Tensor,view操作后的Tensor仍共享存储区
                常见函数:
                    tensor.view(dim1,-1,dimN)   不需要指定每一维度,将其中的一个维度指定为-1,Pytorch会自动计算形状
                    tensor.view_as(other)
                    tensor.squeeze()            去掉Tensor中维度为1维度 (1,3,1,4)变(3,4)
                    tensor.flatten()
                    tensor[None]与tensor.unsqueeze(dim) 为tensor新建一个维度,该维度尺寸为1 (2,3)变(2,1,3)
                
            3.转置
                transpose vs permute
                    transpose,用于两个维度的转置     
                    permute,对任意高维矩阵进行转置  
                    注意:
                        tensor.t() tensor.T tensor.transpose()均为tensor.permute()的特例
                        若Tensor的维度数据排列与输入一致,则使用reshape,否则transpose
                        
                大多数转置操作的输入与输出共享Tensor的存储区,但是转置操作使Tensor在内存空间中变得不连续,tensor.contiguous()将其变连续
                部分操作支持对内存空间中不连续的Tensor进行运算,无需连续化操作,节省内存/显存
    """

    """
    2.线性回归
        y = wx+b+e  
            x,y是输入输出数据  w,x是可学习的参数  误差e~N(0,σ^2)
        loss = ....
            sgd更新w和b来最小化损失函数,最终获取w和b的值
    """

    """
    3.autograd和计算图基础
        1.autograd用法,requires_grad与backward
            1.torch.autograd根据输入和前向传播自动构建计算图,执行反向传播
                autograd记录与网络相关的所有Tensor操作,形成一个前向传播的有向无环图DAG:
                    输入网络的Tensor称为叶子节点,网络输出的Tensor称为根节点
                    从根节点遍历,对所有requires_grad=True的Tensor进行求导,逐层遍历至叶子节点,通过链式操作计算梯度,自动完成反向传播操作
            
            2.torch.autograd.backward(
                tensor=t.tensor([1, 2, 3]), 计算梯度的tensor,torch.autograd.backward(y) 等价于 y.backward()
                grad=t.tensor([1, 2, 3]),   等价于链式法则中的dz/dy(dz/dx = dz/dy * dy/dx)
                retain_graph=True,          反向传播缓存的一些中间结果,反向传播后,缓存就会被清空,通过此参数不清空函数,用于多次反向传播
                create_graph=True)          对反向传播过程再次构建计算图
            
            3.LeafTensor
                计算图中的叶子节点,通常不会直接使用非叶子节点的梯度信息,LeafTensor的设计初衷用于节约内存/显存,
                LeafTensor的判断:
                    Tensor.requires_grad = False
                    Tensor.requires_grad = True 且Tensor由用户创建,保留梯度信息
                LeafTensor.grad_fn=None
        
        2.autograd原理,计算图
            autograd底层采用计算图,有向无环图,记录算子和变量之间的关系
                算子,MUL ADD
                变量,w x b
            1.Tensor.requires_grad 
            2.Tensor.grad_fn 查看Tensor的反向传播函数
              Tensor.grad_fn.next_functions 保存grad_fn的输入,tuple类型
            3.计算w梯度时,需要用到x的值(∂y/∂w = x),该值在前向传播过程中会被保存为buffer(在反向传播过程中不需要更新),反向传播后会被自动清空
              为能进行多次反向传播,需指定retain_graph=True保存这些不需要更新的值 
            4.Tensor的requires_grad默认为False,如果一个节点的requires_grad设为True,则依赖它的所有节点的requires_grad均为True
              有时并不希望autograd对Tensor求导,因为求导存需缓存中间结果,增加额外的内存/显存开销
              对不需要反向传播的场景,关闭自动求导即可 
            5.想要修改Tensor的值,又不希望被autograd记录,可对tensor.data进行操作
            6.反向传播种,非叶子节点的梯度不会被保存,若想查看这些变量的梯度,有两种方法:
                autograd.grad()
                hook() 推荐
            7.Pytorch中计算图的特点:
                1.autograd根据用户对Tensor的操作来建计算图,这些操作可以被抽象为Function
                2.用户创建的节点被称为叶子节点,叶子节点的grad_fn=None
                  对于在叶子节点中需要求导的Tensor,因为其梯度是累加的，所以具有 AccumulateGrad 标记
                3.Tensor默认是不需要求导的,即requires_grad=False
                  如果某个节点的requires_grad=True,则所有依赖它的节点的requires_grad=True
                4.在多次反向传播中,梯度是不断累加的
                  反向传播过程中的中间缓存仅在当次反向传播时有效,为进行多次反向传播,需指定retain_graph=True来保留存档的计算图
                5.在反向传播过程棋中,非叶子节点的梯度不会被保存,可以使用autograd.grad或hook() 来获取非叶子节点的梯度
                6.Tensor.grad与Tensor.data的形状一致,应避免直接修改tensor.data.因为对data的直接操作无法利用autograd进行反向传播
                7.PyTorch采用动态图设计,用户可以很方便地查看中间层的输出,从而动态地设计计算图结构
        
        3.扩展autograd,Function
            1.绝大多数函数可以使用autograd实现反向求导
              若实现一个复杂函数,但不支持自动反向求导,则写个Function,实现前向传播和反向传播的代码       
            
            2.自定义Function
                继承torch.autograd.Function,实现的forward()和backward()属于静态方法
                backward()的输出和forward()的输入一一对应,backward()的输入和forward()的输出一一对应
                反向传播时,可能会利用前向传播的某些中间结果,在前向传播过程中,需保存这些中间结果,否则前向传播结束后这些对象即被释放
                使用Function.apply(tensor)调用新实现的Function
                
        4.利用autograd实现线性回归      
            每次进行反向传播前,需将梯度清零,避免累加
    """