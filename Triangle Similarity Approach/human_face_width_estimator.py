# Calculate Face esitmations


weight = 74.00
height = 179.832
age = 21
sex = 'M'
faceWidthIndexAdult = [6 + (5 / 8), 6 + (3 / 4), 6 + (7 / 8), 7 + (1 / 8), 7 + (1 / 4), 7 + (3 / 8), 7 + (1 / 2),
                       7 + (5 / 8), 7 + (3 / 4), 7 + (7 / 8), 8, 8 + (1 / 8)]
faceWidthIndexSize = ["XS", "S", "S", "M", "M", "L", "L", "XL", "XL", "XXL", "XXL", "XXXL", "XXXL"]
faceWidthIndexYouthSize = ["S", "S", "M", "M", "L", "L", "XL", "XL", "XXL", "XXL"]
faceWidthIndexYouth = [6 + (1 / 8), 6 + (1 / 4), 6 + (3 / 8), 6 + (1 / 2), 6 + (5 / 8), 6 + (3 / 4), 6 + (7 / 8), 7]
small = []
medium = []
large = []
extraLarge = []
doubleExtraLarge = []
superDoubleLarge = []

height_axis = [150, 155, 160, 165, 170, 175, 180, 185]
weight_axis = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
sizeChart = [
    ['XS', "S", "S", "M", "M", "L", "L", "L", "XL", "XL", "XL", "XXL", 'XXL', 'XXL', 'NULL', 'NULL'],
    ['XS', "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", "XL", "XXL", 'XXL', 'XXL', 'NULL'],
    ['XS', 'XS', "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", "XL", "XXL", 'XXL', 'XXL'],
    ['XS', 'XS', "S", "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", "XXL", 'XXL', 'XXL'],
    ['XS', 'XS', "S", "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", "XXL", 'XXL', 'XXL'],
    ['XS', 'XS', "S", "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", 'XL', "XXL", 'XXL'],
    ['NULL', 'XS', "S", "S", "S", "M", "M", "M", "L", "L", "L", "XL", "XL", 'XL', "XXL", 'XXL'],

]
i = 0
j = 0
# if(height > 150)
if weight % 5 == 0:
    i = weight
else:
    # print(weight%5)
    i = weight + 5 - weight % 5

if i not in weight_axis:
    print("Sorry your're unique !")

if height % 5 == 0:
    j = height
else:
    j = height + 5 - height % 5

if j not in height_axis:
    print("Sorry your're unique !")

# print(weight,height)
indexOFx = weight_axis.index(i)
indexOFy = height_axis.index(j)

# print(indexOFy)
# print(sizeChart[indexOFy][])
print(indexOFy, indexOFx)
mainSize = sizeChart[indexOFy - 1][indexOFx]
# print( sizeChart[0][15])
# print(indexOFy,indexOFx)
# print(sizeChart[5][0])
# print(mainSize)
# for i in range(11,19):
#     small.append(i)
# for i in range(11,19):
#     large.append(i)
# for i in range(11,19):
#     extraLarge.append(i)
# for i in range(11,19):
#     doubleExtraLarge.append(i)
# for i in range(11,19):
#     superDoubleLarge.append(i)
multiplier = 0.005
headWidth = 0
if (age < 23):
    size = faceWidthIndexYouthSize.index(mainSize)
    if (sex == 'F'):
        faceWidthIndexYouthSize.index(mainSize, size)
    headWidth = faceWidthIndexYouth[size]
else:
    temp = 0
    temp = abs(age - 20)

    size = faceWidthIndexSize.index(mainSize)
    temp_size = faceWidthIndexSize.index(mainSize, size + 1)
    print(size, temp_size)
    if (sex == 'F'):
        size = min(temp_size, size)
    else:
        size = max(temp_size, size)

    # size =
    print(size)
    headWidth = faceWidthIndexAdult[size]
print(headWidth)
print(mainSize)
temp = 1
final = headWidth + headWidth * multiplier * temp
print("Multipler =", str(headWidth * multiplier))
print("Weight =", str(weight) + ' kg')
print("Height =", str(height) + ' cm')
print("Age =", str(age))
print("Gender =", sex)
print("Estimated Face Width =", final)
