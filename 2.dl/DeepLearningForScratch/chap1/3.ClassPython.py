class Man:
    def __init__(self,name):
        self.name = name
        print('Initialized')

    def hello(self):
        print('hello'+self.name+"!")

    def goodbye(self):
        print('goodbye'+self.name+"!")

if __name__ == '__main__':
    man = Man('David')
    man.hello()
    man.goodbye()