import numpy as np
from multiprocessing import Manager
import concurrent.futures
import logging
import os
import time
from PIL import Image
import re
import matplotlib.pyplot as plt


# Файл для логгирования
log_file = 'data_circles_first_try.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')


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


def init_static_arrays():
    """Инициализация глобальных массивов импульсов и релаксации."""

    global time_relax, cond_relax, cond_imp, time_imp

    file1 = r'calibration.txt'
    file2 = r'calibration_relax.txt'

    try:
        logging.info(f"Попытка загрузить файл импульсов: {file1}")
        cond_imp, time_imp = read_arrays_from_file(file1, 1)
        logging.info(f"Файл импульсов успешно загружен. Размеры array1_imp: {len(cond_imp)}, array2_imp: {len(time_imp)}")

        logging.info(f"Попытка загрузить файл релаксации: {file2}")
        cond_relax, time_relax = read_arrays_from_file(file2, 0)
        logging.info(f"Файл релаксации успешно загружен. Размеры array1_relax: {len(cond_relax)}, array2_relax: {len(time_relax)}")

    except Exception as e:
        logging.error(f"Ошибка при загрузке массивов: {str(e)}")
        raise e
    
    p1 = 255 / cond_imp[-1]
    p2 = 255 / cond_relax[0]

    cond_imp = cond_imp * p1
    cond_relax = cond_relax * p2

    plt.plot(time_imp, cond_imp)
    plt.savefig('imp_hekk.pdf')

    plt.plot(time_relax, cond_relax)
    plt.savefig('rel_hekk.pdf')

    logging.info(f"Массивы импульсов и релаксации успешно загружены и нормализованы. Размеры нормализованных массивов: array1_imp: {len(cond_imp)}, array1_relax: {len(cond_relax)}")

    return time_relax, cond_relax, cond_imp, time_imp


def find_closest_value(arr, target):
    """
    Находит индекс ближайшего элемента в массиве к целевому значению.
    param: arr - массив значений
    param: target - целевое значение
    returns: индекс ближайшего элемента в массиве
    """
    if len(arr) == 0:
        logging.error("Массив пуст, невозможно найти ближайшее значение.")
        raise ValueError("Массив пуст.")

    differences = np.abs(arr - target)
    closest_index = np.argmin(differences)

    return closest_index


def impulse(conductivity, cond_imp, time_imp, time_b):
    """Функция для расчета проводимости после импульса."""
    if conductivity is None:
        logging.error(f"Ошибка: conductivity равен None")
        raise ValueError("Conductivity is None")

    if cond_imp is None or time_imp is None:
        logging.error("Массивы импульса не загружены корректно")
        raise ValueError("Impulse arrays are None")

    time_index = find_closest_value(cond_imp, conductivity)
    dt = time_imp[time_index]

    A = -4.90752341e+00
    B = 1.61927247e+00
    C = -3.46762146e-03
    D = 2.71757824e-06

    up = A + B * (dt + time_b) + C * pow(dt + time_b,2) + D * pow(dt+time_b,3)
    
    return min(up, 255)


def relax(conductivity, cond_relax, time_relax, time_b):
    """Функция для расчета проводимости после релаксации."""
    if conductivity is None:
        logging.error(f"Ошибка: conductivity равен None")
        raise ValueError("Conductivity is None")

    if cond_relax is None or time_relax is None:
        logging.error("Массивы релаксации не загружены корректно")
        raise ValueError("Relax arrays are None")

    time_index = find_closest_value(cond_relax, conductivity)
    dt = time_relax[time_index]

    A = 2.33142078e+02
    B = 2.96052741e-04
    C = 3.96503308e+01
    D = 2.96035007e-04

    down = (A * np.exp(B * (-(time_b + dt))) + C * np.exp(D * (-(time_b + dt))))
    
    return max(down, 0.00000000001)


def get_sorted_image_paths(directory_path):
    """Функция для считывания и сортировки файлов из директории"""
    image_paths = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(directory_path, filename))

    image_paths.sort(key=lambda x: int(re.search(r'^(\d+)', os.path.basename(x)).group(1)))

    return image_paths


# def save_object_image(all_struct, height, width, folder_path, name):
#     """Функция для сохранения изображения с текущими значениями Conductivity."""
#     image = np.zeros((height, width), dtype=np.uint8)

#     for i in range(height):
#         for j in range(width):
#             image[i][j] = all_struct.get(f"{i}_{j}", 0)

#     img = Image.fromarray(image, 'L')

#     if isinstance(name, int):
#         name = str(name)

#     if not os.path.splitext(name)[1]:
#         name += '.png'

#     img.save(f"{folder_path}/{name}")


def save_object_image(all_struct, height, width, folder_path, name):
    """Функция для сохранения изображения с текущими значениями Conductivity."""
    image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel_key = f"{i}_{j}"
            if pixel_key not in all_struct:
                logging.error(f"Значение Conductivity отсутствует для пикселя с координатами ({i}, {j}). Прекращение выполнения.")
                raise ValueError(f"Значение Conductivity не найдено для пикселя с координатами ({i}, {j}).")

            image[i][j] = all_struct[pixel_key]

    # image = np.clip(image, 0, 255).astype(np.uint8)

    img = Image.fromarray(image, 'L')

    if isinstance(name, int):
        name = str(name)

    if not os.path.splitext(name)[1]:
        name += '.png'

    img.save(f"{folder_path}/{name}")


def extract_coordinates_from_path(path):
    """код просто достает имя файла из пути к нему"""
    filename = os.path.basename(path)
    parts = filename.split('_')
    
    if len(parts) < 2:
        logging.error(f"Неправильный формат имени файла: {filename}")
        return None
    
    try:
        return filename
    
    except Exception as e:
        logging.error(f"Ошибка при извлечении координат из файла {filename}: {str(e)}")
        return None


def init_structures_worker(args):
    """Создание словаря проводимости пикселей нейроморфного сенсора"""
    time_array, pow1, coef, one_event_time, height, width, start_row, end_row, all_struct, cond_imp, time_imp, cond_relax, time_relax = args

    for i in range(start_row, end_row):
        for j in range(0, width):
            key = f"{i}_{j}"
            conductivity = 0 
            all_struct[key] = conductivity


def process_image_worker(args):
    """Вычисление реакции нейроморфного сенсора на словарь проводимости"""
    width, height, start_row, end_row, object_figure_path, time_b, all_struct, cond_imp, time_imp, cond_relax, time_relax = args

    with Image.open(object_figure_path).convert('L') as im:
        im = np.array(im)

        if np.all(im == 0):
            logging.warning(f"Изображение {object_figure_path} полностью черное, пропускаем обработку.")
            return  

        for i in range(start_row, end_row):
            for j in range(width):

                key = f"{i}_{j}"
                if key in all_struct:
                    cond = all_struct[key]
                    if cond is None:
                        logging.error(f"Conductivity is None at {key}")
                        raise ValueError(f"Conductivity is None at {key}")


                    if im[i][j] != 0:
                        new_cond = impulse(cond, cond_imp, time_imp, time_b)
                        # logging.debug(f"Обновленная проводимость (impulse) пикселя {key}: {new_cond}")
                        all_struct[key] = new_cond
                    else:
                        new_cond = relax(cond, cond_relax, time_relax, time_b)
                        # logging.debug(f"Обновленная проводимость (relax) пикселя {key}: {new_cond}")
                        all_struct[key] = new_cond


if __name__ == '__main__':


    init_static_arrays()


    start_time = time.time()
    logging.info("Запуск программы")

    dir_path = 'data_circles'
    time_b = int(input("Input time between moves: "))

    print(f'Total number of CPU: {os.cpu_count()}')
    n_processes = int(input("number of using CPU: "))

    path_array = get_sorted_image_paths(dir_path)
    shape_1 = Image.open(path_array[0]).convert('L')
    shape_1 = np.array(shape_1)
    height, width = shape_1.shape
    logging.info(f"Размер изображения: {height}x{width}")

    pow1 = 8
    coef = 1.7
    time_array = np.arange(5, 51)

    folder_path_image = 'classic_small_sensor_image'
    folder_path_figure = 'classic_small_sensor_folder'

    os.makedirs(folder_path_image, exist_ok=True)
    os.makedirs(folder_path_figure, exist_ok=True)

    with Manager() as manager:
        all_struct = manager.dict()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            rows_per_process = height // n_processes
            args_list = [
                (time_array, pow1, coef, time_b, height, width, i * rows_per_process, 
                 (i + 1) * rows_per_process if i < n_processes - 1 else height, all_struct, cond_imp, time_imp, cond_relax, time_relax)
                for i in range(n_processes)
            ]

            futures = [executor.submit(init_structures_worker, args) for args in args_list]
            
            for future in futures:
                try:
                    future.result() 
                except Exception as e:
                    logging.error(f"Ошибка в процессе создания структур: {str(e)}")

        logging.info(f"Первичная обработка завершена. Количество ключей в all_struct: {len(all_struct)}")

        for path in path_array:

            logging.info(f"Обработка файла: {path}")
            name = extract_coordinates_from_path(path)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                args_list = [
                    (height, width, i * rows_per_process, (i + 1) * rows_per_process if i < n_processes - 1 else height, 
                     path, time_b, all_struct, cond_imp, time_imp, cond_relax, time_relax)
                    for i in range(n_processes)
                ]

                futures = [executor.submit(process_image_worker, args) for args in args_list]

                for future in futures:
                    try:
                        future.result()  
                    except Exception as e:
                        logging.error(f"Ошибка в процессе обработки изображения {name}: {str(e)}")

            try:
                save_object_image(all_struct, height, width, folder_path_image, name)
                logging.info(f"Изображение {name} сохранено успешно.")
            except Exception as e:
                logging.error(f"Не удалось сохранить изображение {name}: {str(e)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Время выполнения программы: {elapsed_time} секунд")
    print(f"Время выполнения: {elapsed_time} секунд")


# def save_object_image(all_struct, height, width, folder_path, name):
#     """Функция для сохранения изображения с текущими значениями Conductivity."""
#     image = np.zeros((height, width), dtype=np.uint8)

#     for i in range(height):
#         for j in range(width):
#             image[i][j] = all_struct.get(f"{i}_{j}", 0)

#     img = Image.fromarray(image, 'L')

#     if isinstance(name, int):
#         name = str(name)

#     if not os.path.splitext(name)[1]:
#         name += '.png'

#     img.save(f"{folder_path}/{name}")


# def extract_coordinates_from_path(path):
#     filename = os.path.basename(path)
#     parts = filename.split('_')
#     x = parts[-2]
#     y = parts[-1].split('.')[0]

#     return filename


# if __name__ == '__main__':
#     # Инициализация статических массивов
#     init_static_arrays()

#     start_time = time.time()
#     logging.info("Запуск программы")

#     dir_path = 'classic_sensor'
#     time_b = int(input("Input time between moves: "))
#     logging.info(f"Пользователь ввел время между движениями: {time_b}")

#     print(f'Total number of CPU: {os.cpu_count()}')
#     n_processes = int(input("del for max number of cpu: "))

#     path_array = get_sorted_image_paths(dir_path)
#     shape_1 = Image.open(path_array[0]).convert('L')
#     shape_1 = np.array(shape_1)
#     height, width = shape_1.shape
#     logging.info(f"Размер изображения: {height}x{width}")

#     pow1 = 8
#     coef = 1.7
#     time_array = np.arange(5, 51)

#     folder_path_image = 'classic_small_sensor_image'
#     folder_path_figure = 'classic_small_sensor_folder'
#     os.makedirs(folder_path_image, exist_ok=True)
#     os.makedirs(folder_path_figure, exist_ok=True)

#     with Manager() as manager:
#         all_struct = manager.dict()

#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             rows_per_process = height // n_processes
#             args_list = [
#                 (time_array, pow1, coef, time_b, height, width, i * rows_per_process,
#                  (i + 1) * rows_per_process if i < n_processes - 1 else height, all_struct)
#                 for i in range(n_processes)
#             ]

#             futures = [executor.submit(init_structures_worker, args) for args in args_list]

#             for future in futures:
#                 try:
#                     future.result()  # Проверяем выполнение каждой задачи
#                 except Exception as e:
#                     logging.error(f"Ошибка в процессе создания структур: {str(e)}")

#         logging.info(f"Первичная обработка завершена. Количество ключей в all_struct: {len(all_struct)}")

#         for path in path_array:
#             logging.info(f"Обработка файла: {path}")
#             name = extract_coordinates_from_path(path)

#             with concurrent.futures.ProcessPoolExecutor() as executor:
#                 args_list = [
#                     (height, width, i * rows_per_process, (i + 1) * rows_per_process if i < n_processes - 1 else height,
#                      path, time_b, all_struct)
#                     for i in range(n_processes)
#                 ]

#                 futures = [executor.submit(process_image_worker, args) for args in args_list]

#                 for future in futures:
#                     try:
#                         future.result()  # Проверяем выполнение каждой задачи
#                     except Exception as e:
#                         logging.error(f"Ошибка в процессе обработки изображения {name}: {str(e)}")

#             try:
#                 save_object_image(all_struct, height, width, folder_path_image, name)
#                 logging.info(f"Изображение {name} сохранено успешно.")
#             except Exception as e:
#                 logging.error(f"Не удалось сохранить изображение {name}: {str(e)}")

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     logging.info(f"Время выполнения программы: {elapsed_time} секунд")
#     print(f"Время выполнения: {elapsed_time} секунд")
