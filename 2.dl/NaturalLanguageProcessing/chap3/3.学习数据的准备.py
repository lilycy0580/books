
from chap3.common.util import preprocess,create_contexts_target,convert_one_hot

if __name__ == '__main__':
    if __name__ == '__main__':
        # 语料预处理
        text = 'You say goodbye and I say hello.'
        corpus, word_to_id, id_to_word = preprocess(text)
        print(corpus)
        print(word_to_id)
        print(id_to_word)
        """
            [0 1 2 3 4 1 5 6]
            {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
            {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}    
        """

        # 获取预料上下文和目标
        contexts, target = create_contexts_target(corpus, window_size=1)
        print(contexts)
        print(target)
        """
        [[0 2]
         [1 3]
         [2 4]
         [3 1]
         [4 5]
         [1 6]]
        [1 2 3 4 1 5]
        """

        # 转为one-hot表示
        vocab_size = len(word_to_id)
        target = convert_one_hot(target, vocab_size)
        contexts = convert_one_hot(contexts, vocab_size)
        print(contexts)
        print(target)
        """
        contexts:
            [[[1 0 0 0 0 0 0]
              [0 0 1 0 0 0 0]]

             [[0 1 0 0 0 0 0]
              [0 0 0 1 0 0 0]]

             [[0 0 1 0 0 0 0]
              [0 0 0 0 1 0 0]]

             [[0 0 0 1 0 0 0]
              [0 1 0 0 0 0 0]]

             [[0 0 0 0 1 0 0]
              [0 0 0 0 0 1 0]]

             [[0 1 0 0 0 0 0]
              [0 0 0 0 0 0 1]]]

        target:      
            [[0 1 0 0 0 0 0]
             [0 0 1 0 0 0 0]
             [0 0 0 1 0 0 0]
             [0 0 0 0 1 0 0]
             [0 1 0 0 0 0 0]
             [0 0 0 0 0 1 0]]
        """