import numpy as np


arr = [4, 5, 4]
print(arr)

final_array = []
def insert(index):

    if index < 0:
        return

    if index == 0:
        if index not in final_array and arr[index] > 0:
            final_array.append(index)
        return

    if arr[index] <= 0:
        return

    if index - 1 not in final_array:
        final_array.append(index)
        return
    else:
        if arr[index-1] < arr[index]:
            remove(index-1)
            final_array.append(index)

    return

def remove(index):

    if index in final_array:
        final_array.remove(index)
        insert(index - 1)

    return

def printFinal():
    output = [arr[x] for x in final_array]
    print(output, final_array)

for i,a in enumerate(arr):
    insert(i)

printFinal()

final_array = []
arr = arr[1:]
for i,a in enumerate(arr):
    insert(i)
printFinal()

