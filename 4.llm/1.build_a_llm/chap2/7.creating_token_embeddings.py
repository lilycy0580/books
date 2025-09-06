
import torch as t
"""
    创建词元嵌入  将词元ID转换为嵌入向量   
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # demo:词元ID转换为嵌入向量 仅包含6个单词的词汇表,创建维度为3的嵌入
    input_ids = t.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3
    embedding_layer = t.nn.Embedding(vocab_size, output_dim)    # 嵌入层
    print(embedding_layer.weight)
    """
    Parameter containing:
    tensor([[ 0.3374, -0.1778, -0.1690],
            [ 0.9178,  1.5810,  1.3010],
            [ 1.2753, -0.2010, -0.1606],
            [-0.4015,  0.9666, -1.1481],
            [-1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096]], requires_grad=True)   
    注:
        嵌入层是由随机值构成,在模型训练中不断优化 6*3
        每一行对应词汇表中的一个词元,每一列对应一个嵌入维度
    """

    # 1.根据单个词元ID获取嵌入向量
    embedding_vector = embedding_layer(t.tensor([3]))
    print(embedding_vector)
    """
    tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
    注:
        词元ID=3的嵌入向量与嵌入矩阵中第4行完全相同 嵌入层执行查找操作,根据词元ID从嵌入层的权重矩阵中检索出相应的行
    """

    # 2.根据4个输入ID获取嵌入向量
    print(embedding_layer(input_ids))
    """
    tensor([[ 1.2753, -0.2010, -0.1606],
            [-0.4015,  0.9666, -1.1481],
            [-2.8400, -0.7849, -1.4096],
            [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
    """

