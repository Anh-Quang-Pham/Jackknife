#!/usr/bin/python

import argparse
#import matplotlib.pyplot as plt
import numpy as np
import math

inf = float('inf')

def getParser():
    import os
    import subprocess
    string = subprocess.check_output(['stty', 'size'])
    col = int(string.split()[1])
    os.environ["COLUMNS"] = "{0}".format(int(0.9*col))
    usage = "Usage:..."
    return argparse.ArgumentParser(prog = 'jackknife', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

def logicChecksOnArguments(args):

#    print ("Checking command line options...")
    if args.nbin < 0 or args.ndat < 0:
        print("Arguments [ndat,nbin] are ALL expected to be positive integers! Abort...")
        exit(-1)
    if args.nbin >= args.ndat:
        print("There cannot be more bins than data and the bin length has to be at least 1. Abort...")
        exit(-1)
    if (args.ndat%args.nbin)/float(args.ndat) >= 0.10:
        print("Your are discarding more than 10% of your dataset! Consider changing nbin! Abort...")
        exit(-1)
    if args.execType == 'kU' or args.execType == 'sK' or args.execType == 'sU':
        import os
        if not os.path.exists(args.datafile):
            print("The file '{}', that you would want to read in data from, does NOT exist! Abort...".format(args.datafile))
            exit(-1)
    return

def addGeneralArguments(parser):
    parser.add_argument('--ndat', type = int, default = 1000, dest = 'ndat', help = 'The number of data to read from the datafile')
    parser.add_argument('--nbin', type = int, default = 10, dest = 'nbin', help = 'The number of bins')
    parser.add_argument('--vol', type = int, help = 'The lattice size', dest = 'NS')

def addKurtosisArguments(parser):
    parser.add_argument('--kurtosis', type = str, help = 'Kurtosis function relating primary to secondary observable', dest = 'kur')
    parser.add_argument('--datafile', type = str, default = 'infile', help = 'Path to file containing data', dest = 'datafile')

def addSkewnessArguments(parser):
    parser.add_argument('--skewness', type = str, help = 'Skewness function relating primary to secondary observable', dest = 'ske')
    parser.add_argument('--datafile', type = str, default = 'infile', help = 'Path to file containing data', dest = 'datafile')

def addSusceptibilityArguments(parser):
    parser.add_argument('--susceptibility', type = str, help = 'Susceptibility function relating primary to secondary observale', dest = 'sus')
    parser.add_argument('--datafile', type = str, default = 'infile', help = 'Path to file containing data', dest = 'datafile')

def parseOptions(parser):
#    print("Parsing command line options...")
    subparsers = parser.add_subparsers(title = 'Specific jackknife executables', description = 'Valid options to run specific jakknife executables', help = 'OPTION -h for additional help for specific jackknife executable', dest = 'execType')

    parserKur = subparsers.add_parser('kU', help = 'Analyse kurtosis: Relies on the existence of a one-column datafile containing the Monte Carlo history for a given observable. Each line corresponds to a measurement of the wanted observable on a given gauge configuration.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    addGeneralArguments(parserKur)
    addKurtosisArguments(parserKur)

    parserSke = subparsers.add_parser('sK', help = 'Analyse skewness: Relies on the existence of a one-column datafile containing the Monte Carlo history for a given observable. Each line corresponds to a measurement of the wanted observable on a given gauge configuration.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    addGeneralArguments(parserSke)
    addSkewnessArguments(parserSke)

    parserSus = subparsers.add_parser('sU', help = 'Anayse susceptibility: Relies on the existence of a one-column datafile contaiting the Monte Carlo history for a given observable. Each line corresponds to a measurement of the wanted observable on a given gauge configuration.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    addGeneralArguments(parserSus)
    addSusceptibilityArguments(parserSus)

# Checks lines in infile for the keyword, saves the by \t separated following value
def readOutObservablesBasedOnKeyword(keyword, infile):
    values = np.array([])
    for line in open(infile):
        if keyword in line:
            splitted_line = line.split('\t')
            for i in range(0, len(splitted_line)):
                if keyword == splitted_line[i]:
                    values = np.append(values, float(splitted_line[i + 1]))
    return values

# Construct resampled datasets by deleting sequences of length L
# First N%L data points are cut at the beginning
# Returns, in an array, the overall average followed by the average 
# for each subsample consisting of all, but the k-th, bins
def binData(array):
    N = args.ndat
    L = N/args.nbin
    return np.array([np.mean(array[N%L:])] + [np.mean(np.delete(array[N%L:], np.s_[L*k:L*(k + 1)], 0)) for k in range(args.nbin)])

# Calculates the bias corrected average value of obs and its jackknife error
# Input is the array of averages produced by the binData function
def jackknife(obs, binvals):
    N_blocks = len(binvals) - 1
    if N_blocks < 1:
        print("jackknife: you passed empty or single values array!")
        av = inf
        err = inf
    else:
        av = binvals[0]
        err = 0.
        for k in range(1, N_blocks + 1, 1):
            err += pow(binvals[k] - av, 2)
        err = np.sqrt((N_blocks - 1.0)/N_blocks*err)
        # bias correction
        av = N_blocks*av - (N_blocks - 1.0)*np.mean(binvals[1:])
        print("%s %.6e\t%.6e" % (obs, av, err))
        return av, err

# Evaluate the function given as command line option on x
def kurtosisFunction(x, x2, x3, x4):
    return (x4 - 4.*x3*x + 6.*x2*x**2 - 3.*x**4)/pow(x2 - x**2,2)

def skewnessFunction(x, x2, x3):
    return (x3 - 3.*x2*x + 2.*x**3)/pow(x2 - x**2, 3./2.)

def susceptibilityFunction(x, x2):
    return (x2 - x**2)*pow(args.NS,3)

def setupNeededArraysForkU(args):
    p = np.zeros((args.nbin + 1))
    p2 = np.zeros((args.nbin + 1))
    p3 = np.zeros((args.nbin + 1))
    p4 = np.zeros((args.nbin + 1))
    secondary = np.zeros((args.nbin + 1))
    error = np.zeros((args.nbin + 1))
    return p, p2, p3, p4, secondary

def setupNeededArraysForsK(args):
    p = np.zeros((args.nbin + 1))
    p2 = np.zeros((args.nbin + 1))
    p3 = np.zeros((args.nbin + 1))
    secondary = np.zeros((args.nbin + 1))
    error = np.zeros((args.nbin + 1))
    return p, p2, p3, secondary

def setupNeededArraysForsU(args):
    p = np.zeros((args.nbin + 1))
    p2 = np.zeros((args.nbin + 1))
    secondary = np.zeros((args.nbin + 1))
    error = np.zeros((args.nbin + 1))
    return p, p2, secondary

def analyseKurtosis(args, p, p2, p3, p4, secondary):
#    print("Jackknife-ing kurtosis...")
    kU = np.loadtxt(args.datafile, unpack = True)
    p = binData(kU)
    p2 = binData(kU**2)
    p3 = binData(kU**3)
    p4 = binData(kU**4)
    secondary = kurtosisFunction(p, p2, p3, p4)
#    print("#B4\t dB4")
    value, error = jackknife("", secondary)
    return value, error

def analyseSkewness(args, p, p2, p3, secondary):
#   print("Jackknife-ing skewness...")
    kU = np.loadtxt(args.datafile, unpack = True)
    p = binData(kU)
    p2 = binData(kU**2)
    p3 = binData(kU**3)
    secondary = skewnessFunction(p, p2, p3)
#    print("#B3\t dB3")
    value, error = jackknife("", secondary)
    return value, error

def analyseSusceptibility(args, p, p2, secondary):
#    print("Jackknife-ing susceptibility...")
    kU = np.loadtxt(args.datafile, unpack = True)
    p = binData(kU)
    p2 = binData(kU**2)
    secondary = susceptibilityFunction(p, p2)
#    print("#ChiL\t dChiL")
    value, error = jackknife("", secondary)
    return value, error

def writeOutFile(args, secondary, error):
    if args.execType == 'kU':
        outfile = open("dataFile_B4", 'w')
        outfile.write("%.6e\t%.6e\n" % (secondary, error))
        outfile.close()
    elif args.execType == 'sK':
        outfile = open("dataFile_B3", 'w')
        outfile.write("%.6e\t%.6e\n" % (secondary, error))
        outfile.close()
    elif args.execType == 'sU':
        outfile = open("dataFile_ChiL", 'w')
        outfile.write("%.6e\t%.6e\n" % (secondary, error))
        outfile.close()

if __name__ == "__main__":
    parser = getParser()
    parseOptions(parser)
    args = parser.parse_args()
    logicChecksOnArguments(args)
    if args.execType == 'kU':
        p, p2, p3, p4, secondary = setupNeededArraysForkU(args)
        value, error = analyseKurtosis(args, p, p2, p3, p4, secondary)
        writeOutFile(args, value, error)
    elif args.execType == 'sK':
        p, p2, p3, secondary = setupNeededArraysForsK(args)
        value, error = analyseSkewness(args, p, p2, p3, secondary)
        writeOutFile(args, value, error)
    elif args.execType == 'sU':
        p, p2, secondary = setupNeededArraysForsU(args)
        value, error = analyseSusceptibility(args, p, p2, secondary)
        writeOutFile(args, value, error)
    exit(0)

