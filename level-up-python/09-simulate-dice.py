from collections import defaultdict
import random

def roll_dice(*args, num_trials=100000):
    result = defaultdict(int)

    for _ in range(num_trials):
        result[sum([random.randint(1, x) for x in args])]+=1

    print("OUTCOME PROBABILITY")
    for k,v in dict(sorted(result.items())).items():
        print(f"{k:<7} {v/num_trials*100:.2f}%")


def roll_dice1(*args):

    def get_counts(*args):
        current_dice=args[0]
        if len(args)==1:
            return {i: 1 for i in range(1, current_dice + 1)}
        else:
            new_dict = defaultdict(int)
            for i in range(1, current_dice+1):
                for k, v in get_counts( *args[1:] ).items():
                    new_dict[k+i]+=v
            return new_dict

    print("OUTCOME PROBABILITY")
    total=1
    for number in args:
        total *= number
    for k,v in get_counts(*args).items():
        print(f"{k:<7} {v/total*100:.5f}%")

# commands used in solution video for reference
if __name__ == '__main__':
    roll_dice(1,2,3,4,5,6,7,8,9,10)
    roll_dice(4, 6, 6)
    roll_dice(4, 6, 6, 20)
