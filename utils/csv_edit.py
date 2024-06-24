import csv

input_file = '../graphs/accountNormUsage.csv'
output_file = 'normUsageDict.csv'

with open(input_file, 'r') as file:
    lines = file.readlines()

lines = [lines[0]] + [line.lstrip() for line in lines[1:]]

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for line in lines:
        writer.writerow(line.strip().split(','))

print(f"Finishing writing file to {output_file}")
