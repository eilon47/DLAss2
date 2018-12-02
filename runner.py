from optparse import OptionParser
import numpy as np
from tagger1 import routine

option_parser = OptionParser()
option_parser.add_option("-t", "--type", dest="type", help="Choose the type of the tagger", default="pos")
option_parser.add_option("-e", "--embedding", help="if you want to use the pre trained embedding layer", dest="E", default=False, action="store_true")
option_parser.add_option("-f", "--fix", help="if you want to use prefix and suffix embedding", dest="F", default=False, action="store_true")
option_parser.add_option("-p", "--plot", help="if you want to show plots", dest="plot", default=False, action="store_true")
option_parser.add_option("-l", "--lr", dest="lr", help="Choose the learning rate", default=None, type=float)
option_parser.add_option("-i", "--iter", dest="epochs", help="Choose the epochs", default=None, type=int)


if __name__ == '__main__':
    options, args = option_parser.parse_args()
    for t in ["pos", "ner"]:
        options.type = t
        for epoch in range(1,5):
            options.epochs = epoch
            for lr in np.linspace(0.01, 0.12, 6):
                options.lr = lr
                for embedded in [True, False]:
                    options.E = embedded
                    for fix in [True, False]:
                        options.F = fix
                        print options
                        d = routine(options)
                        print "finished options" , options
                        print "\n"