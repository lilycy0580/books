
from chap2.common.util import *
from chap2.data import ptb
from sklearn.utils.extmath import randomized_svd

if __name__ == '__main__':
    """
    单词的分布式表示:
        每个单词表示为固定长度的密集向量
        使用语料库,计算上下文中的单词数量,将其转化为PPMI矩阵,基于SVD降维获取好的单词向量   
    """

    window_size = 2
    wordvec_size = 100

    # 加载数据
    corpus, word_to_id, id_to_word = ptb.load_data('train') # corpus:10000
    vocab_size = len(word_to_id)

    # 计算共现矩阵,ppmi,svd
    print('counting  co-occurrence ...')
    C = create_co_matrix(corpus, vocab_size, window_size)

    print('calculating PPMI ...')
    W = ppmi(C, verbose=True)

    print('calculating SVD ...')
    try:
        # truncated SVD (fast!)
        U, S, V = randomized_svd(W,                             # 输入矩阵,通常是高维数据 特征矩阵等
                                 n_components=wordvec_size,     # 指定保留的主成分(奇异向量)数量,降维后的维度
                                 n_iter=5,                      # 随机算法的迭代次数
                                 random_state=40)               # 随机种子,用于复现结果
    except ImportError:
        # SVD (slow)
        U, S, V = np.linalg.svd(W)                              # U:(10000, 100)
    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

