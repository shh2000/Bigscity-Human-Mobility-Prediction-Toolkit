from random import randint, random


def trans_foursquare_tky(origin):
    month_key = {'Jan': '01', 'Feb': '02', 'Mar': '03',
                 'Apr': '04', 'May': '05', 'Jun': '06',
                 'Jul': '07', 'Aug': '08', 'Sep': '09',
                 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    origin.readline()
    lines = origin.readlines()
    id2lines = {}
    cnt = 0
    for line in lines:
        info = line.replace('\n', '').split(',')
        index = info[0]
        if index not in id2lines.keys():
            id2lines[index] = []
        id2lines[index].append(cnt)
        cnt += 1
    # print(id2lines)
    df = {'type': 'FeatureCollection', 'features': []}
    for index in id2lines.keys():
        item = {'type': 'Feature', 'properties': {}}
        item['properties']['uid'] = index
        item['properties']['share_extend':str] = {}  # some extend info for this traj

        geometry = {'type': 'Polygon', 'coordinates': []}
        for length in id2lines[index]:
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
            month = month_key[month]
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
        index = info[1].replace(' ', '')
        long = info[3]
        if '.' not in long:
            cnt += 1
            continue
        if index not in id2lines.keys():
            id2lines[index] = []
        id2lines[index].append(cnt)
        cnt += 1
    # print(id2lines)
    df = {'type': 'FeatureCollection', 'features': []}
    for index in id2lines.keys():
        item = {'type': 'Feature', 'properties': {}}
        item['properties']['uid'] = index
        item['properties']['share_extend':str] = {}  # some extend info for this traj

        geometry = {'type': 'Polygon', 'coordinates': []}
        for length in id2lines[index]:
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
        index = lines[i].split('\t')[0]
        index = int(index)
        # print(id)
        if index != last_id:
            id2lines[last_id] = (last_start_line, i)
            last_id = index
            last_start_line = i
    print('finish pre')
    df = {'type': 'FeatureCollection', 'features': []}
    last_line = 0
    for index in id2lines.keys():
        if id2lines[index][0] > last_line + 300000:
            print(id2lines[index][0])
            last_line = id2lines[index][0]
        item = {'type': 'Feature', 'properties': {}}
        item['properties']['uid':str] = index
        item['properties']['share_extend':str] = {}  # some extend info for this traj

        geometry = {'type': 'Polygon', 'coordinates': []}
        for length in range(id2lines[index][0], id2lines[index][1], 1):
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
        locs[i:int] = (float(locs[i].split(' ')[0]), float(locs[i].split(' ')[1]))

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
        times[i:int] = (int(year), int(month), int(day), int(hour), int(minute), int(second))
    df['type'] = 'FeatureCollection'
    df['features'] = []
    item = {'type': 'Feature', 'properties': {}}
    item['properties']['uid':str] = 1
    item['properties']['share_extend':str] = {}  # some extend info for this traj

    geometry = {'type': 'Polygon', 'coordinates': []}
    for length in range(len(locs)):
        node = {'location': [locs[length][0], locs[length][1]], 'time_format': ['111111']}
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
    assert origin == 'BIGSCITY'
    num = 10
    df = {'type': 'FeatureCollection', 'features': []}
    for index in range(num):
        item = {'type': 'Feature', 'properties': {}}
        item['properties']['uid':str] = index + 1
        item['properties']['share_extend':str] = {}  # some extend info for this traj

        geometry = {'type': 'Polygon', 'coordinates': []}
        for length in range(randint(4, 15)):
            node = {'location': [gen_long_lati(), gen_long_lati()], 'time_format': ['111111'],
                    'time': ['2020-02-01-13-59-32'], 'index': length, 'solid_extend': {}}
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    return df
