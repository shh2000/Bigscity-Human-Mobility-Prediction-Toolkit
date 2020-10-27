from datasets.basic import Dataset, RELATIVE_PATH
import json
from random import randint, random


def trans_foursquare_tky(origin):
    monthkey = {'Jan': '01', 'Feb': '02', 'Mar': '03'
        , 'Apr': '04', 'May': '05', 'Jun': '06'
        , 'Jul': '07', 'Aug': '08', 'Sep': '09'
        , 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    origin.readline()
    lines = origin.readlines()
    id2lines = {}
    cnt = 0
    for line in lines:
        info = line.replace('\n', '').split(',')
        id = info[0]
        if id not in id2lines.keys():
            id2lines[id] = []
        id2lines[id].append(cnt)
        cnt += 1
    # print(id2lines)
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    for id in id2lines.keys():
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = id
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for length in id2lines[id]:
            node = {}
            line = lines[length]
            long = float(line.split(',')[5])
            lati = float(line.split(',')[4])
            venue_id = line.split(',')[1]
            node['location'] = [long, lati]
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            time = line.split(',')[-1].replace('\n', '')
            day = time.split(' ')[2]
            year = time.split(' ')[-1]
            hms = time.split(' ')[3].replace(':', '-')
            month = time.split(' ')[1]
            month = monthkey[month]
            time = year + '-' + month + '-' + day + '-' + hms
            node['time'] = [time]
            node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = {'venue_id': venue_id}  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    return df


def trans_foursquare(origin):
    origin.readline()
    origin.readline()
    lines = origin.readlines()
    id2lines = {}
    cnt = 0
    for line in lines:
        if '|' not in line:
            break
        info = line.replace('\n', '').split('|')
        id = info[1].replace(' ', '')
        long = info[3]
        if '.' not in long:
            cnt += 1
            continue
        if id not in id2lines.keys():
            id2lines[id] = []
        id2lines[id].append(cnt)
        cnt += 1
    # print(id2lines)
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    for id in id2lines.keys():
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = id
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for length in id2lines[id]:
            node = {}
            line = lines[length]
            long = float(line.split('|')[3].replace(' ', ''))
            lati = float(line.split('|')[4].replace(' ', ''))
            venue_id = int(line.split('|')[2].replace(' ', ''))
            node['location'] = [long, lati]
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            time = line.split('|')[-1][1:]
            time = time.replace(' ', '-')
            time = time.replace(':', '-')
            time = time.replace('\n', '')
            node['time'] = [time]
            node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = {'venue_id': venue_id}  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    return df


def trans_gowalla(origin):
    lines = origin.readlines()
    # lines = lines[0:456860]
    id2lines = {}

    last_id = 0
    last_start_line = 0
    for i in range(len(lines)):
        id = lines[i].split('\t')[0]
        id = int(id)
        # print(id)
        if id != last_id:
            id2lines[last_id] = (last_start_line, i)
            last_id = id
            last_start_line = i
    print('finish pre')
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    last_line = 0
    for id in id2lines.keys():
        if id2lines[id][0] > last_line + 300000:
            print(id2lines[id][0])
            last_line = id2lines[id][0]
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = id
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for length in range(id2lines[id][0], id2lines[id][1], 1):
            node = {}
            line = lines[length]
            long = float(line.split('\t')[2])
            lati = float(line.split('\t')[3])
            loc_id = int(line.replace('\n', '').split('\t')[-1])
            node['location'] = [long, lati]
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            time = line.split('\t')[1]
            time = time.replace('T', '-')
            time = time.replace(':', '-')
            time = time.replace('Z', '')
            node['time'] = [time]
            node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = {'loc_id': loc_id}  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    return df


def trans_sample(origin):
    df = {}

    origin.readline()
    origin_s = ''
    for line in origin.readlines():
        origin_s += line
    locs = origin_s.split('(')[1].split(')')[0]
    locs = locs.split(',')
    for i in range(1, len(locs), 1):
        locs[i] = locs[i][1:]
    for i in range(len(locs)):
        locs[i] = (float(locs[i].split(' ')[0]), float(locs[i].split(' ')[1]))
    """print(len(locs))
    print(locs)"""

    times = origin_s.split('|')[6]
    times = times.split(';')
    for i in range(len(times)):
        ymd = times[i].split(' ')[0]
        hms = times[i].split(' ')[1]
        year = float(ymd.split('-')[0])
        month = float(ymd.split('-')[1])
        day = float(ymd.split('-')[2])
        hour = float(hms.split(':')[0])
        minute = float(hms.split(':')[1])
        second = float(hms.split(':')[2])
        times[i] = (int(year), int(month), int(day), int(hour), int(minute), int(second))
    """print(times)
    print(len(times))"""
    df['type'] = 'FeatureCollection'
    df['features'] = []
    item = {}
    item['type'] = 'Feature'
    item['properties'] = {}
    item['properties']['uid'] = 1
    item['properties']['share_extend'] = {}  # some extend info for this traj

    geometry = {}
    geometry['type'] = 'Polygon'
    geometry['coordinates'] = []
    for length in range(len(locs)):
        node = {}
        node['location'] = [locs[length][0], locs[length][1]]
        node['time_format'] = ['111111']  # 6bits represent year2second
        # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
        now = times[length]
        time_now = str(now[0])
        for i in range(5):
            time_now += '-'
            time_now += str(now[i + 1])
        node['time'] = [time_now]
        node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
        node['solid_extend'] = {}  # some extend info for this node
        geometry['coordinates'].append(node)

    item['geometry'] = geometry
    df['features'].append(item)
    return df


def gen_long_lati():
    o = random()
    o = o - 0.5
    o *= 360
    return o


def trans_format(origin):
    num = 10
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    for index in range(num):
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = index + 1
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for length in range(randint(4, 15)):
            node = {}
            node['location'] = [gen_long_lati(), gen_long_lati()]
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            node['time'] = ['2020-02-01-13-59-32']
            node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = {}  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    return df


map = {'foursquare-tky': trans_foursquare_tky, 'foursquare': trans_foursquare, 'gowalla': trans_gowalla,
       'sample': trans_sample, 'format': trans_format}


class Translater(Dataset):
    def load(self, dataset_name):
        origin = open(RELATIVE_PATH + '/datasets/cache/' + dataset_name + '.csv')
        df = map[dataset_name](origin)
        return df
