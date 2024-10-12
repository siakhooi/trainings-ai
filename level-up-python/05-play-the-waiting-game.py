import random
import time
def waiting_game():
    target = random.randrange(1,10)
    print(f"Your target time is {target} seconds")

    input("---Press Enter to Begin---")
    start_time=time.time()

    input(f"...Press Enter again after {target} seconds...")
    stop_time=time.time()

    elapsed_time=stop_time-start_time
    diff_time=target-elapsed_time

    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    if diff_time==0:
        print("(Unbelievable! Perfect Timing!)")
    elif diff_time>0:
        print(f"({diff_time:.3f} seconds too fast)")
    else:
        print(f"({-diff_time:.3f} seconds too slow)")

# commands used in solution video for reference
if __name__ == '__main__':
    waiting_game()
