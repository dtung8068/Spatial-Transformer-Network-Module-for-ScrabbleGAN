with open('imgs/cvl_labels.txt', 'r') as file:
    arr = file.read().split()
print(len(max(arr, key=len)))
