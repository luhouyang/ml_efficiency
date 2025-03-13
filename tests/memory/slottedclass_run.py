from slottedclass_metrics import basecls, slottedcls, datacls, timetest

if __name__ == '__main__':
    _ = timetest()

    test = 0

    if test == 0:
        # control value function
        _ = basecls()

    elif test == 1:
        # test value function
        _ = slottedcls()

    elif test == 2:
        _ = datacls()
