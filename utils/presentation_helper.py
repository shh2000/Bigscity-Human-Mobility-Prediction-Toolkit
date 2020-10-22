'''
存放 pre 阶段可能具有共通性的函数
'''
import geohash2
from datetime import datetime, timedelta

'''
使用 geohash2 来直接 encode 经纬度
'''
def encodeLoc(loc):
    return geohash2.encode(loc[0], loc[1]) # loc[0]: latitude, loc[1]: longtitude 

'''
将 json 中 time_format 格式的 time 转化为 datatime
'''
def parseTime(time, time_format):
    '''
    parse to datetime
    '''
    # only implement 111111
    if time_format[0] == '111111':
        return datetime.strptime(time[0], '%Y-%m-%d-%H-%M-%S') # TODO: check if this is UTC Time ?

'''
用于切分轨迹成一个 session
思路为：给定一个 start_time 找到一个基准时间 base_time，在该 base_time 到 base_time + time_length 区间的点划分到一个 session 内
选取 base_time 来做的理由是：这样可以保证同一个小时段总是被 encode 成同一个数
'''
def calculateBaseTime(start_time, base_zero):
    if base_zero:
        return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)

'''
计算两个时间之间的差值，返回值以小时为单位
'''
def calculateTimeOff(now_time, base_time):
    # 先将 now 按小时对齐
    now_time = now_time - timedelta(minutes=now_time.minute, seconds=now_time.second)
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600
