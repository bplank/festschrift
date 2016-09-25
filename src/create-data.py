import sys

user=0
train = []
train_y = []
test = []
test_y = []
dev = []
dev_y = []
header = open(sys.argv[1]).readlines()[0].strip().split("\t")[2:]

for line in open(sys.argv[1]).readlines()[1:]: #ignore header
    line=line.strip()
    f = line.split("\t")
    session = f[1]
    if session in ["1","2"]: # use first two sessions as dev
        dev.append([i for i in f[2:]])
        dev_y.append(int(f[0]))
    elif session in ["3","4","5"]: # use three sessions as test
        test.append([i for i in f[2:]])
        test_y.append(int(f[0]))

    else:
        train.append([i for i in f[2:]])
        train_y.append(int(f[0]))

OUTD=open("{}.dev".format(sys.argv[1]),"w")
OUTT=open("{}.test".format(sys.argv[1]),"w")
OUTTR=open("{}.train".format(sys.argv[1]),"w")

OUTT.write("user\t{}\n".format("\t".join(header)))
for t,y in zip(test,test_y):
    OUTT.write("{}\t{}\n".format(y,"\t".join(t)))
OUTT.close()

OUTTR.write("user\t{}\n".format("\t".join(header)))
for t,y in zip(train,train_y):
    OUTTR.write("{}\t{}\n".format(y,"\t".join(t)))
OUTTR.close()

OUTD.write("user\t{}\n".format("\t".join(header)))
for t,y in zip(dev,dev_y):
    OUTD.write("{}\t{}\n".format(y,"\t".join(t)))
OUTD.close()

