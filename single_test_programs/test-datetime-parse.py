from datetime import datetime

def parse_datetime_string(string):
    dt_format = '%Y-%m-%d %H:%M:%S'
    date_format = '%Y-%m-%d'
    time_format = '%H:%M:%S'
    try:
        parsed_dt = datetime.strptime(string, dt_format)
    except ValueError:
        pass
    else:
        return parsed_dt, 'datetime'

    try:
        parsed_dt = datetime.strptime(string, date_format)
    except ValueError:
        pass
    else:
        return parsed_dt, 'date'

    try:
        parsed_dt = datetime.strptime(string, time_format)
    except ValueError:
        return string, 'none'
    else:
        return parsed_dt, 'time'


print(parse_datetime_string('2018-09-03'))

print(parse_datetime_string('14:30:3'))

print(parse_datetime_string('2015-12-03 3:15:0'))

print(parse_datetime_string('boom'))