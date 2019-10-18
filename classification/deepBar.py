def barBegin(num, title='epoch'):
    print('%s: %d' % (title, num))

def barRun(current, total, show_num, title='loss'):
    squ_num = int(20*(current/total))
    print('\r{0}{1}{2}'.format('▉ '*squ_num, title+': ', show_num), end='')

def barEnd():
    print('')