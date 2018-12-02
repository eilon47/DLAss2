######################## Authors ###########################

Daniel Greenspan, 308243948
Eilon Bashari, 308576933

###########################################################



################################## WELCOME ########################################

First of all we to make it clear that all the taggers are available from tagger1.py file.
the tagger1.py file uses the tagger1_utils.py as it's utilities file.
This program also reads it's configuration from json file named "taggers_params.json" which you can change at every moment.
This program uses flags to indicate which options we want to use on the tagger (was taked from the help on it):
        Options:
            -h, --help                show this help message and exit
            -t TYPE, --type=TYPE      Choose the type of the tagger 'ner' or 'pos'
            -e, --embedding           If you want to use the pre trained embedding matrix
            -f, --fix                 If you want to use prefix and suffix embedding matrices
            -p, --plot                If you want to show plots
            -l LR, --lr=LR            Choose the learning rate
            -i EPOCHS, --iter=EPOCHS  Choose the epochs

######################################################################################



################# Commands for execution for each part ###############################

Part 1:
to run tagger1 which is not pre-trained and is not using the prefix and suffix embedding matrices please run the program with the following command:
        python tagger1.py -t <type>
The result will be found in file "test1.<type>"

Part 3:
to run tagger2 which is pre trained but not using the prefix and suffix embedding please run the program with the following command:
        python tagger1.py -t <type> -e
The result will be found in file "test3.<type>"

Part 4:
to run tagger3 which is pre trained and using the prefix and suffix embedding please run the program with the following command:
        python tagger1.py -t <type> -e -f
The result will be found in file "test4.<type>"

you can always run part 4 without using the embedded matrix with the command : python tagger1.py -t <type> -f

#######################################################################################



###### Please Notice #####

The command lines above does not show the plot of the program, if you want to see it please add the flag "-p"

you can always change the learning rate and the number of epochs in the configuration file or using the flags :
    -l <LR> or --lr=<LR> , -i <EPOCHS> or --iter=<EPOCHS>

##########################



#####################  Requirements #########################

Numpy, Pytorch, Python-tk, Matplotlib

#############################################################


