# -*- coding: utf-8 -*-

#This program does the following-
#There are n containers with elements from two arrays a and b
#Sort the containers with respect to elements of array a
#return the array b by retrieving elements serially from the sorted containers

class Relative_Sorter:
    def __init__(self, array):
        self.array = array
        self.indices = [i for i in range(len(self.array))]
    
    def sort(self):
        for i in range(len(self.array)): 
            min_idx = i 
            for j in range(i+1, len(self.array)): 
                if self.array[min_idx] > self.array[j]: 
                    min_idx = j        
            self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]
            self.indices[i], self.indices[min_idx] = self.indices[min_idx], self.indices[i]
        
    def getRelative(self, in_arr):
        ret_arr = []
        for i in range(len(self.indices)):
            ret_arr.append(in_arr[self.indices[i]])
        return ret_arr

if __name__ == "__main__":
    a = [3.24, 1.11, 2.89, 5.47, 4.36]
    b = [4.88, -2.65, 9.67, 7.4, -4.81]
    sorter = Relative_Sorter(a)
    sorter.sort()
    b = sorter.getRelative(b)
    print(b)
