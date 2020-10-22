import geohash2


'''
使用 geohash2 来直接 encode 经纬度
'''
def encode_loc(loc):
    return geohash2.encode(loc[0], loc[1])  # loc[0]: latitude, loc[1]: longtitude