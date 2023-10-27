import manage_list as ml
import sorting as s
import time as t
from math import log10

def merge_worst_case(A):
    n = len(A)
    if n > 1:
        m = n // 2
        left = [A[i*2] for i in range(n-m)]
        right = [A[i*2+1] for i in range(m)]
        A = merge_worst_case(left) + merge_worst_case(right)
    return A

def insertion_worst_case(sorting_list):
    for i in range(1, len(sorting_list)):
        j = i
        while sorting_list[j - 1] < sorting_list[j] and j>0:
            sorting_list[j - 1], sorting_list[j] = sorting_list[j], sorting_list[j - 1]
            j -= 1

def tim_merge_worste_case(sorting_list):
    n = len(sorting_list)
    if n > 1:
        m = n // 2
        left = [sorting_list[i*2] for i in range(n-m)]
        right = [sorting_list[i*2+1] for i in range(m)]
        sorting_list = merge_worst_case(left) + merge_worst_case(right)
    return sorting_list

def tim_worst_case(sorting_list, step = 32):
    chunks = [sorting_list[i:i + step] for i in range(0, len(sorting_list), step)]
    chunks = merge_worst_case(chunks)
    for element in chunks:
        insertion_worst_case(element)
    chunks = [inner for outer in chunks for inner in outer]
    return chunks

def insertion_sort(word_list):
    comp_swap = 0
    beginning = t.time()
    for i in range(1, len(word_list)):
        j = i
        while word_list[j - 1] > word_list[j] and j>0:
            comp_swap += 1
            word_list[j - 1], word_list[j] = word_list[j], word_list[j - 1]
            comp_swap += 2
            j -= 1
    ending = t.time()
    time = ending - beginning
    return comp_swap, time

def main():
    print("--------------------------------------------------------------------------\n")
    #limit = 2*int(input("Bis zu welcher Potenz von 10 sollen die Listen groß werden? "))
    #for i in [i*0.5 for i in range(limit)]:
    #    print(int(10**i))
    #    for j in range(5):
    #        print(i, j)
    #        ml.create_list(int(10**i))
    #        A = ml.get_list("list.txt")
    #        average_rec = s.sort_rec(A.copy())
    #        average_loop = s.sort_loop(A.copy())
    #        average_tim = s.sort_other(A.copy())
    #        ml.save("average_rec.txt", int(10**i), average_rec)
    #        ml.save("average_loop.txt", int(10**i), average_loop)
    #        ml.save("average_tim.txt", int(10**i), average_tim)
    #for i in [i*0.5 for i in range(limit)]:
    #    print(int(10**i))
    #    for j in range(5):
    #        print(i, j)
    #        A = [k for k in range(int(10**i))]
    #        best_rec = s.sort_rec(A.copy())
    #        best_loop = s.sort_loop(A.copy())
    #        best_tim = s.sort_other(A.copy())
    #        ml.save("best_rec.txt", int(10**i), best_rec)
    #        ml.save("best_loop.txt", int(10**i), best_loop)
    #        ml.save("best_tim.txt", int(10**i), best_tim)
    #for i in [i*0.5 for i in range(limit-3)]:
    #    print(int(10**i))
    #    for j in range(5):
    #        print(i, j)
    #        A = [k for k in range(int(10**i))]
    #        worst_rec = s.sort_rec(merge_worst_case(A))
    #        worst_loop = s.sort_loop(merge_worst_case(A))
    #        worst_tim = s.sort_other(tim_worst_case(A))
    #        ml.save("worst_rec.txt", int(10**i), worst_rec)
    #        ml.save("worst_loop.txt", int(10**i), worst_loop)
    #        ml.save("worst_tim.txt", int(10**i), worst_tim)
    #for i in [i*0.5 for i in range(10, 21)]:
    #i=4.75
    #print(int(10**i))
    #for j in range(3):
    #    print(i, j)
    #    ml.create_list(int(10**i))
    #    A = ml.get_list("list.txt")
    #    B = A.copy()
    #    print(j, "check1")
    #    average_insertion = insertion_sort(B)
    #    print(j, "check2")
    #    ml.save("insertion.txt", int(10**i), average_insertion)
    #    print(j, "check3")
    #    average_tim = s.sort_other(A)
    #    print(j, "check4")
    #    ml.save("insertion_tim.txt", int(10**i), average_tim)
    #    print(j, "check5")
    #for i in [i*0.5 for i in range(11)]:
    #    for j in range(5):
    #        s.progress_bar((i*5)+j, 180)
    #        ml.create_list(int(10**i))
    #        A = ml.get_list("list.txt")
    #        average_loop = s.sort_loop(A.copy())
    #        average_tim = s.sort_other(A)
    #        ml.save("average_tim.txt", int(10**i), average_tim)
    #        ml.save("average_loop.txt", int(10**i), average_loop)
    #for i in [i*0.5 for i in range(10)]:
    #    for j in range(5):
    #        s.progress_bar((i*5)+j+55, 180)
    #        A = [i for i in range(int(10**i))]
    #        worst_loop = s.sort_loop(merge_worst_case(A))
    #        worst_tim = s.sort_other(tim_merge_worste_case(A))
    #        ml.save("worst_tim.txt", int(10**i), worst_tim)
    #        ml.save("worst_loop.txt", int(10**i), worst_loop)
    #for i in [i*0.5 for i in range(15)]:
    #    for j in range(5):
    #        s.progress_bar((i*5)+j+105, 180)
    #        A = [i for i in range(int(10**i))]
    #        best_loop = s.sort_loop(A)
    #        best_tim = s.sort_other(A)
    #        ml.save("best_tim.txt", int(10**i), best_tim)
    #        ml.save("best_loop.txt", int(10**i), best_loop)
    #for i in [i*0.5 for i in range(20)]:
    #i=5
    #for j in range(3):
    #    s.progress_bar(j, 3)
    #    ml.create_list(int(10**i))
    #    A = ml.get_list("list.txt")
    #    avg_tim = s.sort_other(A.copy())
    #    avg_insertion = insertion_sort(A)
    #    ml.save("insertion_tim.txt", int(10**i), avg_tim)
    #    ml.save("insertion.txt", int(10**i), avg_insertion)
    #for i in range(1, 7):
    #        s.progress_bar(i, 14)
    #        log= int(10**i)*log10(int(10**i))*0.6
    #        ml.save("nlogn.txt", int(10**i), log)
    for i in range(1, 7):
            s.progress_bar(i+7, 14)
            squared= (int(10**i)**2)*0.0025
            ml.save("squared.txt", int(10**i), squared)

if __name__ == "__main__":
    main()
