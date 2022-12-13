x =['0.01_0.001_500_0.4.npy', 4475.0, [[266.0, 18], [277.0, 13], [273.0, 15], [275.0, 13], [281.0, 11], [271.0, 15], [278.0, 12], [32.0, 100], [273.0, 15], [2.0, 100], [34.0, 100], [283.0, 9], [18.0, 100], [274.0, 14], [263.0, 21], [16.0, 100], [273.0, 15], [286.0, 8], [34.0, 100], [4.0, 100], [276.0, 14], [-78.0, 100], [278.0, 12], [286.0, 8]]]





richtige = [0 for i in range(24)]
gelöst = [0 for i in range(1296)]
rewards = []
for idx_ge, elemx in enumerate(x):
    for idx,elem in enumerate(elemx[2]):
        if elem[1] < 100:
            gelöst[idx_ge] += 1
            rewards.append(elem[0])
            richtige[idx] += 1
#calc reward schnitt
a = 0
for r in rewards:
    a += r
print(a/len(rewards))

maxi = max(gelöst)
print(maxi)
print(gelöst.index(maxi))
print(x[gelöst.index(maxi)])