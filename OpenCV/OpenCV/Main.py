from OpenCV import Tasks_OpenCV

print('1 - найти все orb features точки на изображении')
print('2 - найти все sift features точки на изображении')
print('3 - найти canny edges на изображенни')
print('4 - перевести в grayscale')
print('5 - перевести изорбражение в hsv')
print('6 - отразить изображение по правой границе')
print('7 - отразить изображение по нижней границе')
print('8 - повернуть изображение на 45 градусов')
print('9 - повернуть изображение на 30 градусов вокруг заданной точки')
print('10 - сместить изображение но 10 пикселей вправо')
print('11 - изменить яркость изображения')
print('12 - изменить контрастность изображения')
print('13 - сделать гамма-перобразование')
print('14 - сделать гистограмную эквайлизацию')
print('15 - изменить баланс белого, сделать более "теплую" картинку')
print('16 - изменить баланс белого, сделать более "холодную" картинку')
print('17 - изменить цветовую палитру по заданному шаблону')
print('18 - сделать бинаризацию изображения')
print('19 - найти контуры на бинаризированном изображении')
print('20 - нати контуры на изображении, применив фильтры (Собеля или Лапласиан)')
print('21 - сделать размытие изображения')
print('22 - сделать фильтрацию изображения при помощи Фурье преобразоваия, оставить только быстрые частоты')
print('23 - сделать фильтрацию изображения при помощи Фурье преобразоваия, оставить только медленные частоты')
print('24 - применить операцию эрозии к изображению')
print('25 - применить операцию дилатации к изображению')
print()


number = int(input('Введите число: '))

openCV = Tasks_OpenCV()

if (number == 1):
    openCV.find_orb()
if (number == 2):
    openCV.find_sift()
if (number == 3):
    openCV.find_canny()
if (number == 4):
    openCV.convert_grayscale()
if (number == 5):
    openCV.convert_hsv()
if (number == 6):
    openCV.draw_right()
if (number == 7):
    openCV.draw_botton()
if (number == 8):
    openCV.rotate_image45()
if (number == 9):
    openCV.rotate_image_around_back_point()
if (number == 10):
    openCV.replace_image()
if (number == 11):
    openCV.change_brightness()
if (number == 12):
    openCV.change_contrast()
if (number == 13):
    openCV.gamma_conversion()
if (number == 14):
    openCV.histogram_equalization()
if (number == 15):
    openCV.make_warm()
if (number == 16):
    openCV.make_cool()
if (number == 17):
    openCV.change_palette()
if (number == 18):
    openCV.make_image_binarization()
if (number == 19):
    openCV.find_conturs()
if (number == 20):
    print('1 - Собель')
    print('2 - Лапласиан')
    print()
    number = int(input('Введите число: '))
    if (number == 1):
        openCV.make_sobel_filter()
    else:
        openCV.make_laplacian_filter()
if (number == 21):
    openCV.make_blurry_image()
if (number == 22):
    openCV.make_low_frequencies()
if (number == 23):
    openCV.make_fast_frequencies()
if (number == 24):
    openCV.make_erosion()
if (number == 25):
    openCV.make_dilate()
    
