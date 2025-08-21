import sys

if __name__ == '__main__':
    print(sys.version)

    print(1-2)
    print(4*5)
    print(7/5)
    print(3**2)

    print(type(7))
    print(type(2.718))
    print(type('hello'))

    x=10
    print(x)
    x=100
    print(x)
    y=3.14
    print(x*y)
    print(type(x*y))

    hungry = True
    sleepy = False
    print(type(hungry))
    print(not hungry)
    print(hungry and sleepy)
    print(hungry or sleepy)

    # 列表
    a = [1,2,3,4,5]
    print(a)
    print(len(a))
    print(a[0])
    a[4] = 99
    print(a)

    print(a[0:2])
    print(a[1:])

    # 字典
    me = {'height':180}
    print(me['height'])
    me['weight']=70
    print(me)

    # if
    hungry = True
    if hungry:
        print('I am hungry!')
    else:
        print('I am sleepy!')

    # for循环
    for i in[1,2,3]:
        print(i)

    # 函数
    def hello():
        print('hello')

    hello()


    def hello(object):
        print('hello'+object+'!')

    hello('cat')



