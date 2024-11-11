def Adding_Trigger(data):
    if data.shape[0] == 3:
        for i in range(3):
            data[i][1][28] = 1
            data[i][1][29] = 1
            data[i][1][30] = 1
            data[i][2][29] = 1
            data[i][3][28] = 1
            data[i][4][29] = 1
            data[i][5][28] = 1
            data[i][5][29] = 1
            data[i][5][30] = 1

    if data.shape[0] == 1:
        data[0][1][24] = 1
        data[0][1][25] = 1
        data[0][1][26] = 1
        data[0][2][24] = 1
        data[0][3][25] = 1
        data[0][4][26] = 1
        data[0][5][24] = 1
        data[0][5][25] = 1
        data[0][5][26] = 1
    return data