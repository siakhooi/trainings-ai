import time
import sched

def schedule_function(event_time, event_function, *args):
    s = sched.scheduler(time.time, time.sleep)
    s.enterabs(event_time, 1, event_function, argument=args)
    #s1=time.strftime("%a %b %d %H:%M:%S %Y",time.gmtime(event_time))
    s1=time.asctime(time.localtime(event_time))
    print(f"{event_function.__name__}() scheduled for {s1}")
    s.run()

# commands used in solution video for reference
if __name__ == '__main__':
    schedule_function(time.time() + 1, print, 'Howdy!')
    schedule_function(time.time() + 1, print, 'Howdy!', 'How are you?')
