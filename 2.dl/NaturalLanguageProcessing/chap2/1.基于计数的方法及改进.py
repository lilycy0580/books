import numpy as np
from matplotlib import pyplot as plt


# 1.语料库预处理
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word  # 单词ID列表,word2id列表,id2word列表

# 2.生成共现矩阵
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# 3.计算相似度
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)  # 先标准化向量,后计算向量内积
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)  # eps防止除数为0
    return np.dot(nx, ny)


# 4.对相似度进行排序
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] ' + query)

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count = 0
    for i in (-1 * similarity).argsort():  # 此时相似度已有序 argsort()返回升序的index
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

# 计算点互信息 pmi
def ppmi(C, verbose=False, eps = 1e-8):             # C为共现矩阵
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M


if __name__ == '__main__':
    # 语料预处理
    corpus, word_to_id, id_to_word = preprocess('you say goodbye and i say hello.')
    print(corpus, word_to_id, id_to_word)
    #[0 1 2 3 4 1 5 6]
    # {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
    # {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

    # 生成共现矩阵
    vocab_size = len(word_to_id)
    co_matrix = create_co_matrix(corpus, vocab_size)
    print(co_matrix)
    # [[0 1 0 0 0 0 0]
    #  [1 0 1 0 1 1 0]
    #  [0 1 0 1 0 0 0]
    #  [0 0 1 0 1 0 0]
    #  [0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1]
    #  [0 0 0 0 0 1 0]]

    # 计算相似度 you与i的相似度
    c0 = co_matrix[word_to_id['you']]  # you的单词向量
    c1 = co_matrix[word_to_id['i']]  # i的单词向量
    print(cos_similarity(c0, c1))
    # 0.7071067691154799

    # 相似度排序
    most_similar('you', word_to_id, id_to_word, co_matrix, top=5)
    # [query] you
    #  goodbye: 0.7071067691154799
    #  hello: 0.7071067691154799
    #  i: 0.7071067691154799
    #  and: 0.0
    #  say: 0.0

    # 获取点互信息
    W = ppmi(co_matrix)
    np.set_printoptions(precision=3)
    print(W)
    # [[0.    1.807 0.    0.    0.    0.    0.   ]
    #  [1.807 0.    0.807 0.    0.807 0.807 0.   ]
    #  [0.    0.807 0.    1.807 0.    0.    0.   ]
    #  [0.    0.    1.807 0.    1.807 0.    0.   ]
    #  [0.    0.807 0.    1.807 0.    0.    0.   ]
    #  [0.    0.807 0.    0.    0.    0.    2.807]
    #  [0.    0.    0.    0.    0.    2.807 0.   ]]

    # 基于svd降维
    U, S, V = np.linalg.svd(W)
    print("共现矩阵,    ppmi矩阵,     SVD")
    print(co_matrix[0])     # [0 1 0 0 0 0 0]
    print(W[0])             # [0.    1.807 0.    0.    0.    0.    0.   ]
    print(U[0])             # [-3.409e-01 -1.110e-16 -3.886e-16 -1.205e-01  0.000e+00  9.323e-01    2.226e-16]

    # 稀疏向量变稠密向量  降维即可
    print(U[0][:2])         # [-3.409e-01 -1.110e-16]

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.tight_layout()
    plt.savefig('./1.png')
    plt.show()

