#!/usr/bin/env python
import json
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: ./graph.py [log file]")
    exit(-1)

fname = sys.argv[1]

f = open(fname, 'r')
d = json.load(f)
f.close()

epoch = [x["iteration"] for x in d]
gen_loss = [x["generator/loss"] for x in d]
dis_loss = [x["discriminator/loss"] for x in d]

plt.figure(figsize=(10,8))

plt.plot(epoch, gen_loss, label="Generator")
plt.plot(epoch, dis_loss, label="Discriminator")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.grid(True)

plt.legend(loc=1)
plt.show()

