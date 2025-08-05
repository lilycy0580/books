from chap4.common.most_similar import most_similar
from chap4.common.analogy import analogy
import pickle

if __name__ == '__main__':
    pkl_file = 'cbow_params.pkl'
    # pkl_file = 'skipgram_params.pkl'

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

    # most similar task
    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    """
        [query] you
        we: 0.7255859375
        i: 0.689453125
        your: 0.63525390625
        they: 0.630859375
        anything: 0.58251953125
        
        [query] year
        month: 0.8515625
        summer: 0.7998046875
        week: 0.75732421875
        spring: 0.7431640625
        decade: 0.6689453125
        
        [query] car
        window: 0.62451171875
        luxury: 0.615234375
        cars: 0.60498046875
        truck: 0.60107421875
        auto: 0.57763671875
        
        [query] toyota
        seita: 0.6484375
        chevrolet: 0.62548828125
        digital: 0.619140625
        engines: 0.59423828125
        mills: 0.591796875
    """

    # analogy task
    print('-' * 50)
    analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
    analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
    analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
    analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)

    """
        [analogy] king:man = queen:?
         a.m: 5.71484375
         woman: 5.38671875
         toxin: 4.953125
         downside: 4.8359375
         kid: 4.76953125
        
        [analogy] take:took = go:?
         went: 4.38671875
         were: 4.296875
         came: 4.2421875
         're: 4.2265625
         was: 4.078125
        
        [analogy] car:cars = child:?
         a.m: 7.0859375
         daffynition: 5.29296875
         children: 5.234375
         rape: 5.0703125
         bond-equivalent: 4.671875
        
        [analogy] good:better = bad:?
         rather: 6.13671875
         more: 5.8515625
         less: 5.671875
         greater: 4.421875
         fewer: 4.41015625
    """
