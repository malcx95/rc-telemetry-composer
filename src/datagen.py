import math
import random


def main():
    
    with open("test.csv", 'w') as f:
        for i in range(10000):
            f.write("{},{},{},{}\n".format(
                i/100,
                int((math.sin(i/1000) + 1)*100),
                math.sin(i/1000),
                math.cos(i/1000),
            ))



if __name__ == "__main__":
    main()
