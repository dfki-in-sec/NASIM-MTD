results = [[-102.0, '0.8_0.8_500_0.9.npy'], [36.25, '0.01_0.3_500_0.4.npy'], [-93.41666666666667, '0.1_0.5_10000_0.7.npy'], [47.916666666666664, '0.01_0.1_50000_0.4.npy'], [55.833333333333336, '0.01_0.01_10000_0.7.npy'], [28.916666666666668, '0.01_0.01_100000_0.9.npy'], [61.416666666666664, '0.01_0.01_50000_0.7.npy'], [48.0, '0.01_0.1_10000_0.7.npy'], [-35.833333333333336, '0.01_0.3_10000_0.4.npy'], [2.5, '0.1_0.5_500_0.9.npy'], [32.166666666666664, '0.01_0.01_500_0.7.npy'], [-2.0, '0.01_0.01_10000_0.99.npy'], [24.166666666666668, '0.01_0.01_100000_0.4.npy'], [25.416666666666668, '0.01_0.001_50000_0.7.npy'], [52.25, '0.01_0.001_150000_0.7.npy'], [23.666666666666668, '0.01_0.001_500_0.7.npy'], [50.583333333333336, '0.01_0.01_50000_0.9.npy'], [21.416666666666668, '0.01_0.001_100000_0.9.npy'], [-102.0, '0.1_0.5_150000_0.7.npy'], [19.583333333333332, '0.1_0.5_10000_0.4.npy'], [-2.0, '0.5_0.001_10000_0.9.npy'], [-40.083333333333336, '0.01_0.3_100000_0.7.npy'], [-114.0, '0.3_0.1_500_0.99.npy'], [-26.5, '0.1_0.1_100000_0.4.npy']]





results.sort(key=lambda tup: tup[0])

#print results
for elem in results:
    print(elem)
    
"""
endlist = []
for i in range(24):
    re = results[-i][1]
    print(re)
    re = re.split("_")
    endlist.append([re[0],re[1],re[2],re[3][:-4]])

print(endlist) 
"""
