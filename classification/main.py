from train import train
from test import test

def main():
    model = train('LeNet')
    test(model=model)

if __name__ == '__main__':
    main()