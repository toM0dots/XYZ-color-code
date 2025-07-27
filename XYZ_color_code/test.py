import numpy as np 
import argparse
def func(L):
    lis = []
    for i in range(L):
        lis.append(i)
    return lis

def main():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    parser.add_argument('--length', type=int, nargs='+', default= 100,
                        help='Random seed for reproducibility.')
    print(func())

if __name__ == '__main__':
    main()