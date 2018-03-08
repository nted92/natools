import warnings

import time as t
from datetime import datetime, timedelta, time
import calendar

import tzlocal

import pytz
from pytz import all_timezones_set

import numpy as np


def get_local_timezone_name():
    return str(tzlocal.get_localzone())


def decompose_date_int(date_int):
    """
    :param date_int: yyyyMMdd format (int)
    :return: tuple of length 3 (year, month, day)
    """
    year = date_int // 10000
    month_day = date_int - year * 10000
    month = month_day // 100
    day = month_day - month * 100
    return year, month, day


def convert_date_int_to_datetime_object(date_int):
    """
    :param date_int: yyyyMMdd format (int) 
    :return: datetime.datetime() non localized object
    """
    return datetime(*decompose_date_int(date_int))


def convert_datetime_object_to_date_int(datetime_object):
    """
    :param datetime_object: datetime.datetime() object
    :return: date_int yyyyMMdd format (int) 
    """
    return int(datetime_object.year) * 10000 + datetime_object.month * 100 + int(datetime_object.day)


def check_if_valid_date_int(date_int):
    """
    :param date_int: yyyyMMdd format (int) 
    :return: whether the date_int is valid or not (boolean)
    """
    try:
        convert_date_int_to_datetime_object(date_int)
        return True
    except ValueError:
        return False


def shift_date_int(date_int, offset=1):
    """
    :param date_int: yyyyMMdd format (int) 
    :param offset: amount of days (positive or negative integer, default 1)
    :return: shifted date_int (yyyyMMdd format), and None if invalid input
    """
    try:
        dt = convert_date_int_to_datetime_object(date_int)
        return convert_datetime_object_to_date_int(dt + timedelta(int(offset)))
    except ValueError:
        warnings.warn("the date_int provided is not valid!")
        return None


def count_number_of_days_between_two_date_ints(date_int_1, date_int_2):
    """
    :param date_int_1: yyyyMMdd format (int)
    :param date_int_2: yyyyMMdd format (int)
    :return: unique number of days, including the function's arguments, 
    and None if invalid input
    """
    if check_if_valid_date_int(date_int_1) and \
            check_if_valid_date_int(date_int_2):
        if date_int_1 == date_int_2:
            return 1
        elif date_int_1 > date_int_2:
            return count_number_of_days_between_two_date_ints(date_int_2,
                                                              date_int_1)
        else:
            counter = 2
            while shift_date_int(date_int_1, counter - 1) < date_int_2:
                counter += 1
            return counter
    else:
        warnings.warn("at least one of the provided date_int is not valid!")
        return None


def get_date_int_range(date_int_1, date_int_2):
    """
    :param date_int_1: yyyyMMdd format (int)
    :param date_int_2: yyyyMMdd format (int)
    :return: list of the ordered date_ints from min(date_int_1, date_int_2) 
    to max(date_int_1, date_int_2) included, and None if invalid input
    """
    if check_if_valid_date_int(date_int_1) and check_if_valid_date_int(date_int_2):
        if date_int_1 == date_int_2:
            return [date_int_1]
        elif date_int_1 > date_int_2:
            return get_date_int_range(date_int_2, date_int_1)
        else:
            diff = count_number_of_days_between_two_date_ints(date_int_1,
                                                              date_int_2) - 1
            if diff > 1:
                missing_days = [shift_date_int(date_int_1, i)
                                for i in range(1, diff)]
            else:
                missing_days = []
            return [date_int_1] + missing_days + [date_int_2]
    else:
        warnings.warn("at least one of the provided date_int is not valid!")
        return None


def get_date_int_inner_range(date_int_1, date_int_2):
    """
    :param date_int_1: yyyyMMdd format (int)
    :param date_int_2: yyyyMMdd format (int)
    :return: list of the ordered date_ints from min(date_int_1, date_int_2)
    incremented by one day included, to max(date_int_1, date_int_2), 
    decremented by one day included. Returns None if invalid input
    """
    if check_if_valid_date_int(date_int_1) and check_if_valid_date_int(date_int_2):
        if date_int_1 == date_int_2:
            return []
        elif date_int_1 > date_int_2:
            return get_date_int_range(date_int_2, date_int_1)
        elif shift_date_int(date_int_1, 1) == date_int_2:
            return []
        else:
            return get_date_int_range(shift_date_int(date_int_1, 1),
                                      shift_date_int(date_int_2, -1))
    else:
        warnings.warn("at least one of the provided date_int is not valid!")
        return None


def check_if_consecutive_date_ints(date_int_1, date_int_2):
    """
    :param date_int_1: yyyyMMdd format (int)
    :param date_int_2: yyyyMMdd format (int)
    :return: boolean, and None if invalid input
    """
    if check_if_valid_date_int(date_int_1) and\
            check_if_valid_date_int(date_int_2):
        if date_int_1 > date_int_2:
            return check_if_consecutive_date_ints(date_int_2, date_int_1)
        else:
            return shift_date_int(date_int_1) == date_int_2
    else:
        warnings.warn("at least one of the provided date_int is not valid!")
        return None


def get_datetime_object(year, month, day, hour=0, minute=0, second=0,
                        timezone_name=None):
    """
    :param year: int
    :param month: int
    :param day: int
    :param hour: int
    :param minute: int
    :param second: int
    :param timezone_name: string (optional)
    :return: a datetime.datetime object, localized (i.e. timezone aware) if 
    the timezone_name argument was specified
    """
    datetime_object = datetime(year, month, day, hour, minute, second)
    if timezone_name is not None:
        tz = pytz.timezone(timezone_name)
        datetime_object = tz.localize(datetime_object)
    return datetime_object


def get_datetime_object_from_date_int(date_int, timezone_name=None):
    """
    :param date_int: yyyyMMdd format (int)
    :param timezone_name: string (optional)
    :return: a datetime.datetime object, localized (i.e. timezone aware) if 
    the timezone_name argument was specified
    """
    year, month, day = decompose_date_int(date_int)
    return get_datetime_object(year, month, day, timezone_name=timezone_name)


def check_if_datetime_object_localized(datetime_object):
    """
    :param datetime_object: a datetime.datetime object
    :return: whether it is localized or not (boolean)
    """
    return datetime_object.tzinfo is not None \
           and datetime_object.tzinfo.utcoffset(datetime_object) is not None


def delocalize_datetime_object(datetime_object):
    """
    :param datetime_object: a datetime.datetime object
    :return: the delocalized datetime.datetime object
    """
    return datetime_object.replace(tzinfo=None)


def localize_datetime_object(datetime_object, timezone_name):
    """
    :param datetime_object: a datetime.datetime object
    :param timezone_name: string
    :return: the localized datetime.datetime object, or datetime_object 
    unmodified if the timezone_name is invalid.
    
    If the datetime_object was already localized, it will overwrite 
    the localized information. 
    """
    if timezone_name in all_timezones_set:
        tz = pytz.timezone(timezone_name)
        if check_if_datetime_object_localized(datetime_object):
            datetime_object = delocalize_datetime_object(datetime_object)
        return tz.localize(datetime_object)
    else:
        warnings.warn("the timezone_name is not valid!")
        return datetime_object


def convert_datetime_object_to_timestamp(datetime_object, timezone_name=None):
    """
    :param datetime_object: a datetime.datetime object
    :param timezone_name: string (optional)
    :return: timestamp integer (UNIX Epoch in seconds)
    
    If a valid timezone_name is passed, it will overwrite any existing 
    localized information that the datetime_object might contain. 
    """
    if timezone_name is not None:
        localized_datetime = localize_datetime_object(datetime_object,
                                                      timezone_name)
    else:
        localized_datetime = datetime_object
    timestamp = calendar.timegm(localized_datetime.utctimetuple())
    return timestamp


def convert_timestamp_to_datetime_object(timestamp, timezone_name='UTC'):
    """
    :param timestamp: UNIX Epoch in seconds (int)
    :param timezone_name: string (optional, default UTC)
    :return: a datetime.datetime object
    
    If no valid timezone_name is passed (e.g. None), it outputs a result in local timezone of the machine.
    """
    if timezone_name in all_timezones_set:
        dt = datetime.fromtimestamp(timestamp, tz=pytz.timezone(timezone_name))
    else:
        dt = datetime.fromtimestamp(timestamp)
    return dt


def convert_timestamp_to_formatted_date_string(timestamp, timezone_name='UTC'):
    """
    :param timestamp: UNIX Epoch in seconds (int)
    :param timezone_name: string (optional, default UTC)
    :return: string
    """
    return convert_timestamp_to_datetime_object(timestamp, timezone_name).strftime("%Y-%m-%d %H:%M")


def convert_timestamp_to_date_int(timestamp, timezone_name='UTC'):
    """
    :param timestamp: UNIX Epoch in seconds (int)
    :param timezone_name: string (optional, default UTC)
    :return: yyyyMMdd format (int)
    
    If no valid timezone_name is passed (e.g. None), it outputs a result in local timezone of the machine.
    """
    dt = convert_timestamp_to_datetime_object(timestamp, timezone_name)
    return convert_datetime_object_to_date_int(dt)


def get_current_date_int(timezone_name='UTC'):
    """
    :param timezone_name: string (optional, default UTC)
    :return: yyyyMMdd format (int)

    If no valid timezone_name is passed (e.g. None), it outputs a result in local timezone of the machine.
    """
    return convert_timestamp_to_date_int(timestamp=t.time(),
                                         timezone_name=timezone_name)


def get_date_int_extreme_timestamps(date_int, timezone_name=None):
    """
    :param date_int: yyyyMMdd format (int)
    :param timezone_name: string (optional, default None)
    :return: a tuple of the first and last timestamp corresponding to the 
    localized date that was given, and None if the inout date_int is invalid
    
    If no valid timezone_name is passed to the function (e.g. None), it will return the 
    first timestamp that enters that date on the whole planet (
    Pacific/Kiritimati timezone) and the last timestamp to exit that day (
    Pacific/Niue timezone).
    """
    if check_if_valid_date_int(date_int):
        datetime_object_1 = convert_date_int_to_datetime_object(date_int)
        datetime_object_2 = convert_date_int_to_datetime_object(shift_date_int(date_int, 1))
        if timezone_name not in all_timezones_set:
            timezone_name_1, timezone_name_2 = "Pacific/Kiritimati", "Pacific/Niue"
        else:
            timezone_name_1, timezone_name_2 = timezone_name, timezone_name

        localized_datetime_1 = localize_datetime_object(datetime_object_1,
                                                        timezone_name_1)
        localized_datetime_2 = localize_datetime_object(datetime_object_2,
                                                        timezone_name_2)

        return convert_datetime_object_to_timestamp(localized_datetime_1), \
               convert_datetime_object_to_timestamp(localized_datetime_2)-1
    else:
        warnings.warn("the provided date_int is not valid!")
        return None


def convert_date_int_to_timestamp(date_int, timezone_name='UTC'):
    """
    :param date_int: yyyyMMdd format (int)
    :param timezone_name: string (optional, default UTC)
    :return: the first timestamp corresponding to the localized date that was 
    given, and None if the input date_int is invalid.
    
    If no valid timezone_name is passed to the function (e.g. None), it will return the 
    result in first timezone that enters that date on the whole planet (
    Pacific/Kiritimati timezone)
    """
    return get_date_int_extreme_timestamps(date_int, timezone_name)[0]


def check_if_datetime_object_at_midnight(datetime_object):
    """
    :param datetime_object: datetime.datetime object
    :return: boolean
    """
    if datetime_object.time() == time(0, 0):
        return True
    else:
        return False


def get_formatted_time_around_midnight(hour, minute):
    """
    :param hour: hour of the day (24 hours format)
    :param minute: minute (60 minutes format)
    :return: formatted time (from -12 to 12, 0 being midnight)
    """
    if hour == 12:
        return hour
    else:
        sleep_time = hour + minute/60
        return np.sign(12 - sleep_time) * np.min([24 - sleep_time, sleep_time])


def get_formatted_time_around_midnight_from_datetime_object(datetime_object):
    """
    :param datetime_object: datetime.datetime object, optionally localized (
    i.e. with a timezone)
    :return: formatted time (from -12 to 12, 0 being midnight)
    """
    hour, minute = datetime_object.hour, datetime_object.minute
    return get_formatted_time_around_midnight(hour, minute)


def get_formatted_time_around_midnight_from_timestamp(timestamp,
                                                      timezone_name='UTC'):
    """
    :param timestamp: UNIX Epoch in seconds (int)
    :param timezone_name: string (optional, default UTC)
    :return: formatted time (from -12 to 12, 0 being midnight)
    
    If no valid timezone_name is passed (e.g. None), it outputs a result in local timezone of the machine.
    """
    datetime_object = convert_timestamp_to_datetime_object(timestamp,
                                                           timezone_name)
    return get_formatted_time_around_midnight_from_datetime_object(datetime_object)


def convert_formatted_time_around_midnight_back(formatted_time):
    if formatted_time < 0:
        formatted_time = 24 + formatted_time
    return int(formatted_time), int((formatted_time % 1) * 60)


def get_time_intervals_intersection(t1, t2, t3, t4):
    """
    :param t1: first time interval's lower bound
    :param t2: first time interval's upper bound
    :param t3: second time interval's lower bound
    :param t4: second time interval's upper bound
    :return: intervals' intersection as a tuple, or None if no intersection 
    is found or the input arguments are invalid
    """
    if (t1 > t2) or (t3 > t4):
        return None
    else:
        if t1 <= t3:
            if t2 < t3:
                return None
            else:
                if t2 <= t4:
                    return t3, t2
                else:
                    return t3, t4
        else:
            return get_time_intervals_intersection(t3, t4, t1, t2)


def get_time_range_in_bins(start, end, bin_size):
    """
    :param start: timestamp (unit of time x)
    :param end: timestamp (unit of time x)
    :param bin_size: size of bins (unit of time x)
    :return: list
    """
    return list(range(start, end, bin_size)) + [end]


if __name__ == "__main__":
    print(get_date_int_inner_range(20170923, 20170930))
