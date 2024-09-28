import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import newton
import csv
import os
from object import *
from PIL import Image
from object import *
from object import Figure
import matplotlib.pyplot as plt


file = r'/Users/tortolla/Desktop/model_for_serv - 1/calibration.txt'


def read_arrays_from_file(filename, motion):
    """
    Reads two arrays from file. If motion == 0 - reads arrays only before array1 maximum, else only until maximum
    param: filepath, motion
    return: array1, array2
    """

    array1 = np.array(())
    array2 = np.array(())

    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            if len(values) == 2:
                array1 = np.append(array1, float(values[0]))
                array2 = np.append(array2, float(values[1]))

    if(motion == 1):
        index = array1.argmax(axis=0)
        array1 = array1[:index]
        array2 = array2[:index]
    else:
        index = array1.argmax(axis=0)
        array1 = array1[index:]
        array2 = array2[index:]

    return array1, array2


def over_look(structures, delta, name, width, height, path):

    """
    Function saves image every time frame
    params:
    structures - array of class structure objects
    delta - time frame
    name - name of image
    width - width of image
    height - height of image
    """

    pic_in_time = np.zeros((width, height))
    t = 0
    name_time  = 0
    time = structures["1_1"].time0

    while t <= time:
        for i in range(0, width):
            for j in range(0, height):
                pic_in_time[i][j] = structures[f"{i}_{j}"].cond_in_time(t)

        look = Figure(pic_in_time, width, height)
        look.save(f'{name}_{t}', path)
        print(name_time)
        name_time += delta
        t = t + delta


def convert_to_grayscale(image_path):

    """
    Turns image into gray
    params:
    image_path - путь до изображения
    returns:
    pixel_array - массив в форме: [i][j] элемент соответствует яркости [i][j] пикселя
    pixel_array_1 - тот же массив, но в одну строчку
    """

    image = Image.open(image_path, "r")
    grayscale_image = image.convert('L')
    width, height = grayscale_image.size
    pixel_values = grayscale_image.getdata()
    pixel_array_1 = np.array(pixel_values)
    pixel_array = np.array(pixel_array_1).reshape((height, width))

    return pixel_array, pixel_array_1


def after_look(structures, name, width, height, path):

    """
    Function saves image every time frame
    params:
    structures - array of class structure objects
    delta - time frame
    name - name of image
    width - width of image
    height - height of image
    """
    
    name = name
    pic_in_time = np.zeros((width, height))
    flag = 0

    for value in structures['0_0'].array_of_time:
      
      print(structures[f"{'0'}_{'0'}"].event_array_dict)

      
      for i in range(0, width):
         for j in range(0, height):

            structures[f"{i}_{j}"].impulse_time = value
            pic_in_time[i][j] = structures[f"{i}_{j}"].cond_after_all(structures[f"{i}_{j}"].event_array_dict[f"event_array_dict{flag}"])

      look = Figure(pic_in_time, width, height)
      look.save(f"{name}one_impules_time:{structures['0_0'].array_of_time[flag]}", path)
      flag += 1


def create_empty_csv(file_name):
    """
    creates csv file or rewrites csv
    param: filepath
    """

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([])


def find_closest_value(arr, target):
    """
    finds index of closest element to value in array
    param: array, value
    returns: index
    """
    differences = np.abs(arr - target)
    closest_index = np.argmin(differences)

    return closest_index


def equation(x, A, B, C, D):
    """
    Returns double exponential function from input array
    param: array, A, B, C, D, array
    returns: array
    """

    return A * np.exp(B * (-x)) + C * np.exp(D * (-x))


def read_arrays_from_file(filename, motion):
    """
    Reads two arrays from file. If motion == 0 - reads arrays only before array1 maximum, else only until maximum
    param: filepath, motion
    return: array1, array2
    """

    array1 = np.array(())
    array2 = np.array(())

    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            if len(values) == 2:
                array1 = np.append(array1, float(values[0]))
                array2 = np.append(array2, float(values[1]))

    if(motion == 1):
        index = array1.argmax(axis=0)
        array1 = array1[:index]
        array2 = array2[:index]
    else:
        index = array1.argmax(axis=0)
        array1 = array1[index:]
        array2 = array2[index:]

    return array1, array2


def create_dataframe_from_csv(file_path):
    """
    Makes pandas dataframe from csv file
    param: filepath
    returns: dataframe
    """

    df = pd.read_csv(file_path)

    return df

