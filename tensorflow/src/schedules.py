from src.callbacks import Step

def onetenth_150_175(lr):
    steps = [150, 175]
    lrs = [lr, lr / 5, lr / 50]
    return Step(steps, lrs)
