import numpy as np
from PIL import Image, ImageDraw
import math
from scipy import ndimage
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import newton
import csv
from impulse import *
import impulse
import math


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



"Class of figures - creating of image, adding noise, saving, screen of output"
class Figure(object):

    """
    The input is image width, image length
    """

    def __init__(self, pic, width, height):

        self.pic = pic
        self.height , self.width = self.pic.shape
        self.image = Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)

    
    def right_diag(self, min, max):

        """
        Makes image of right diagonal (in array)
        """

        image = np.random.randint(min, max, (self.width, self.height))
        for i in range(self.width):
            image[i, -i-1] = 255
        
        self.pic = image
        

    def left_diag(self, min, max):

        """
        Makes image of left diagonal (in array)
        """

        image = np.random.randint(min, max, (self.width, self.height))
        for i in range(self.width):
            image[i, i] = 255
        
        self.pic = image


    def add_random_shape(self, length, max, min):

        """
        Add a random shape to the images - the brightness of each pixel is determined in the range max-min
        using randint()
        param:
        length - side length
        max - maximum brightness of pixel
        min - minimum brightness of pixel
        returns:
        pic - array with pixel brightness of image with random figure on it
        ***
        Also this function changes self.pic, so you probably don't change
        """

        image_shape = self.pic.shape

        max_x = image_shape[0] - length
        max_y = image_shape[1] - length

        start_x = np.random.randint(0, max_x)
        start_y = np.random.randint(0, max_y)

        for i in range(start_x, start_x + length):
            for j in range(start_y, start_y + length):
                self.pic[i,j] = np.random.randint(min, max)

        self.pic = self.pic

        return self.pic


    def gauss_start(self, average, sigma):

        """
        Adds normal noise on image
        param:
        average - average of normal distribution
        sigma - sigma of normal distribution

        ***
        Changes self.pic array to self.pic with normal noise
        """

        for i in range(0,self.width):
            for j in range(0,self.height):
                self.pic[i][j] = np.random.normal(average, sigma) 


    def add_round_shape_and_gauss_field(self, radius, center_x, center_y, average, sigma):

        """
    Adds filled circle on the image and Gaussian noise on the rest of the image.
    The circle itself is filled with 255.
    Parameters:
    radius - Radius of the circle.
    center_x, center_y - Coordinates (x, y) of the center of the circle.
    average, sigma - Parameters of the normal distribution for the Gaussian noise.
        """

        x = radius
        y = 0
        error = 1 - x

        while x >= y:
            for i in range(int(center_x - x), int(center_x + x) + 1):
                self.pic[int(center_y + y)][i] = 255
                self.pic[int(center_y - y)][i] = 255

            for i in range(int(center_x - y), int(center_x + y) + 1):
                self.pic[int(center_y + x)][i] = 255
                self.pic[int(center_y - x)][i] = 255

            y += 1
            if error < 0:
                error += 2 * y + 1
            else:
                x -= 1
                error += 2 * (y - x + 1)

        for y in range(self.pic.shape[0]):
            for x in range(self.pic.shape[1]):
                if self.pic[y][x] != 255:
                    self.pic[y][x] = np.clip(np.random.normal(average, sigma), 0, 255)

        return self.pic

    
    def add_round_shape_full_gauss(self, radius, center_x, center_y):

        """
        Adds filled circle with gauss noise on image
        param:
        radius - radius of circle
        center_x - coord. x of center
        center_y - coord. y of center
        avarage, sigma  - params of normal distribution
        """

        x = radius
        y = 0
        error = 1 - x

        while x >= y:

            for i in range(int(center_x - x), int(center_x + x) + 1):
                self.pic[int(center_y + y)][i] = 255
                self.pic[int(center_y - y)][i] = 255

            for i in range(int(center_x - y), int(center_x + y) + 1):
                self.pic[int(center_y + x)][i] = 255
                self.pic[int(center_y - x)][i] = 255

            y += 1
            if error < 0:
                error += 2 * y + 1
            else:
                x -= 1
                error += 2 * (y - x + 1)

        return self.pic


    def add_round_shape_full(self, radius, center_x, center_y):

        """
        Adds filled circle with gauss noise on image except the circle itself is filled with 255.
        Parameters:
        radius - radius of circle
        center_x - coord. x of center
        center_y - coord. y of center
        average, sigma  - parameters of normal distribution for noise
        min_noise, max_noise - minimum and maximum values for noise
        """

        x = radius
        y = 0
        error = 1 - x

        while x >= y:
            for i in range(int(center_x - x), int(center_x + x) + 1):
                self.pic[int(center_y + y)][i] = 255
                self.pic[int(center_y - y)][i] = 255

            for i in range(int(center_x - y), int(center_x + y) + 1):
                self.pic[int(center_y + x)][i] = 255
                self.pic[int(center_y - x)][i] = 255

            y += 1
            if error < 0:
                error += 2 * y + 1
            else:
                x -= 1
                error += 2 * (y - x + 1)

        return self.pic


    def add_square_shape(self, side_length, top_left_x, top_left_y, min, max):

        """
        Adds a square with random.randint() distribution
        param:
        side_length - the side length of the square
        top_left_x - x coordinate of the top left corner of the square
        top_left_y - y coordinate of the top left corner of the square
        max, min - params of random.randint() distribution
        """

        for y in range(top_left_y, top_left_y + side_length):
            for x in range(top_left_x, top_left_x + side_length):
                self.pic[y][x] = 255


    def add_centered_circle(self, radius, height_image):

        self.pic[0][3] = 255
        self.pic[0][4] = 255
        self.pic[0][5] = 255
        self.pic[0][6] = 255
        self.pic[1][4] = 255
        self.pic[1][5] = 255
        self.pic[2][4] = 255
        self.pic[2][5] = 255
        self.pic[3][4] = 255


    

    def view(self):

        """
        Shows image on screen
        param:
        """

        for i in range(0, self.width):
            for j in range(0, self.height):
                if (self.pic[i][j] != 0):
                    self.draw.point((i, j), fill=int(self.pic[i][j]))
                else:
                    color = 0
                    self.draw.point((i, j), fill=color)
                    self.pic[i][j] = color
        self.image.show()


    def save(self, name, path):

        """
        Saves image
        param:
        name - name of file
        path - path to file where user wants to save image
        """

        self.pic = np.nan_to_num(self.pic)
 
        color = 0

        for j in range(0, self.width):
            for i in range(0, self.height):
                if(self.pic[i][j] < 0.0001):
                    color = 0
                else:
                    color = int(self.pic[i][j])
                    self.draw.point((i, j), fill=color)

        file_name = name
        self.image.save(f'{path}/{file_name}.png')


    def im_show(self):

        """
        returns image object from PIL that is made from self.pic array
        param:
        returns:
        image - PIL object
        """

        for i in range(0, 8):
            for j in range(0, 8):
                if(self.pic[i][j] < 1):
                    color = 0
                else:
                    color = int(self.pic[i][j])
                    self.draw.point((i, j), fill=color)

        return self.image

"Structure class - determines the structure, its luminescence, the change in its conductivity with time"

class Structure(object):

    C1 = 1.2
    C2 = 1.2
    tau_1 = 0.14
    tau_2 = 1.35
    k = 0.1


    def __init__(self, name, x, y, pixel_color, pow1, coef, array_of_time, one_event_time):

        self.name = name
        self.coord_x = x
        self.coord_y = y
        self.p1 = pow1
        self.coef = coef
        self.color1 = pixel_color
        self.Conductivity = self.color1
        self.time_dict = {}
        self.event_array_dict = {}
        self.array_of_time = array_of_time

        for i, time_value in enumerate(array_of_time):
          
          time_name = f"time{i}"
          event_array_name = f"event_array_dict{i}"

          setattr(self, f"time{i}", np.arange(0, time_value, 1))
          setattr(self, f"event_array_dict{i}", np.ones(time_value))

          self.time_dict[time_name] = getattr(self, time_name)
          self.event_array_dict[event_array_name] = getattr(self, event_array_name)

        self.file1 = r'calibration.txt'
        self.file2 = r'calibration_relax.txt'

        self.cond = np.array(())

        self.impulse_time = one_event_time
        self.array1_imp, self.array2_imp = read_arrays_from_file(self.file1, 1)
        self.array1_relax, self.array2_relax = read_arrays_from_file(self.file2, 0)

        p1 = 255 / self.array1_imp[-1]
        p2 = 255 / self.array1_relax[0]

        self.array1_imp = self.array1_imp * p1
        self.array1_relax = self.array1_relax * p2


    def Poisson_generator(self, rate, n, time, myseed=False):

        """
        The Poisson_generator function is used to generate a Poisson process with a given frequency (rate) and duration (n) to simulate random events
        param:
        rate - frequancy of Poisson process
        n - duration of the Poisson process
        myseed - initial value for the random number generator
        returns:
        poisson train - array with 0/1 for time duration
        """

        Lt = time

        if myseed:
            np.random.seed(seed=myseed)
        else:
            np.random.seed()

        u_rand = np.random.rand(n, Lt)

        firing_delimeter = 4
        dt = 1
        poisson_train = 1. * (u_rand < rate * (dt / firing_delimeter))
        poisson_train = poisson_train.flatten()
        count  = 0
        for i in poisson_train:
            if i == 1:
                count +=1


        return poisson_train


    def encode_pixel_brightness(self, time):

        """
        Translates pixel brightness into rate param for Poisson generator
        returns:
        spike_train - array with 0/1 for time duration (where 1 is spike, and 0 is relax)
        """

        max_brightness = 255

        rate = math.pow(self.color1 / (max_brightness), self.p1) * self.coef
        n = 1
        spike_train = self.Poisson_generator(rate, n, time)
        spike_list = [i for i, spike in enumerate(spike_train.tolist())]


        return spike_train


    def get_self_event_array(self, time):

        """
        Function that gets self_event_array
        returns
        event_array -  array with 0/1 for time duration (where 1 is spike, and 0 is relax)
        """

        event_array = self.encode_pixel_brightness(time)

        return event_array


    def get_all_of_them(self):

      for key, value in self.event_array_dict.items():

        self.event_array_dict[key] = self.get_self_event_array(len(value))


    def sgo_event(self):

        """
        Function that counts the amount of spikes in self.event_array
        returns:
        num - number of 1( number of spikes ) in event_array
        """

        num = 0
        min_len = 0
        flag = 0
        z = 0
        numer = 0
        delta = 0
        event_array = self.event_array[:6000]
        for i in event_array:
            if(i == 1):
                num += 1
            numer+=1
        return(num)


    def impulse(self, conductivity):

        """
        Impulse function that counts conductivity after one spike ( after 1 in event_array )
        param:
        conductivity - conductivity of structure before spike ( before 1 in event_array )
        returns:
        conductivity - conductivity of structure after spike ( after 1 in event_array )
        """

        if hasattr(impulse, 'find_closest_value'):
            time_index = impulse.find_closest_value(conductivity, self.array1_imp)
            dt = self.array2_imp[time_index]
            
        # вызов функции
        else:
            print("Функция find_closest_value не найдена")


        A = -4.90752341e+00
        B = 1.61927247e+00
        C = -3.46762146e-03
        D = 2.71757824e-06

        up = A + B*(dt + self.impulse_time) + C * pow(dt + self.impulse_time, 2) + D * pow(dt + self.impulse_time, 3)
        if(up <= 255):
            return up
        else:
            return 255


    def relax(self, conductivity, time):
       
       """
       Relaxation function that counts conductivity after one relax event( after 0 in event_array )
       param:
       conductivity - conductivity of structure before spike ( before 0 in event_array )
       returns:
       conductivity - conductivity of structure after spike ( after 0 in event_array )
       """

       time_index = impulse.find_closest_value(conductivity, self.array1_relax)
       time = time

       dt = self.array2_relax[time_index]

       A = 2.33142078e+02
       B = 2.96052741e-04
       C = 3.96503308e+01
       D = 2.96035007e-04

       down = (A * np.exp(B * (-(time+dt))) + C * np.exp(D * (-(time+dt))))

       if(down > 0):
           return down
       else:
           return 0.00000000001


    def get_sequence_length(self, start_index, target_value):

        """
        Gets length of sequance from start_index of array until target value in array
        param:
        start_index - index from that sequence starts
        target_value - value that we are searching in array
        returns:
        length - length of seqeunce
        """

        length = 0
        for i in range(start_index, len(self.event_array)):
            if self.event_array[i] == target_value:
                length += 1
            else:
                break
        return length


    def cond_in_time(self, time):

        """
        Function that calculates structures conductivity in time
        param:
        time - time, when user whants to now conductivity
        returns:
        cond - conductivity in time
        """

        i = 0
        cond = self.color
        event_array = self.event_array[:time]

        "Take the event_array before the desired time"
        while (i < (len(event_array))):

            "If 1, we simply apply the pulse at once"
            if (event_array[i] == 1):
                    cond = self.impulse(cond)
                    i+=1
            else:
                "Have to count how long it takes to relax"

                relax_time = 0

                while (event_array[i] == 0):
                    "If the relaxation time is less than the length of the event_array - just look at the next element and add one to the relaxation time"

                    if ((i + relax_time) < (len(event_array) - 1)):

                        relax_time += 1
                        i += 1

                    else:
                        i+=1
                        break
                if(cond == 'None'):
                    cond  = 0
                cond  = self.relax(cond, relax_time)
                #i += length

        return cond


    def cond_after_all(self, event_array):

        i = 0
        cond = self.color1
        event_array = event_array
        #print(event_array)

        "Take the event_array before the desired time"
        while (i < (len(event_array))):

            "If 1, we simply apply the pulse at once"
            if (event_array[i] == 1):
                    cond = self.impulse(cond)
                    i+=1
            else:
                "Have to count how long it takes to relax"

                relax_time = 0

                while (event_array[i] == 0):
                    "If the relaxation time is less than the length of the event_array - just look at the next element and add one to the relaxation time"

                    if ((i + relax_time) < (len(event_array) - 1)):

                        relax_time += 1
                        i += 1

                    else:
                        i+=1
                        break
                if(cond == 'None'):
                    cond  = 0
                cond  = self.relax(cond, relax_time)
                #i += length

        return cond



