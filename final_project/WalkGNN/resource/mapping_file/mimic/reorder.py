import csv

with open('mmfo.csv', 'r') as f:
    data = f.readlines()

d = data[0].split(',')



with open('mimic_feature_order.csv', 'w+') as file:
    for elem in d[2:]:
        file.write(elem + ',')
        file.write('\n')