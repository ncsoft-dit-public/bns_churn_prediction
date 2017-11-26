#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

exp = 0L
money = 0L
start_time = None   # 시작세션 시간

# 주요지표
play_millis = {}    # 1003번이 발견되면 start_time 업데이트, 1004번이 발견되면 start_time 시간이 있으면 비교하여 diff_millis 값을 yyyymmdd 킷값에 amount 값을 늘린다
exp_amount = {}     # 일자별 경험치 스냅샷 - 누적 획득량을 계속 업데이트하면 끗
money_amount = {}   # 일자별 돈 스냅샷 - 누적획득량을 계속 업데이트하면 끗
logs_amount = {}    # 일 별 로그의 빈도수

LOG_START = 1003      # 2:  log_id
LOG_END = 1004        # 2:  log_id
LOG_EXP = 1016        # 37: use_value1_num:획득량, 41:new_value2_num:누적획득량
LOG_MONEY = 1017      # 38: use_value2_num:획득량, 42:new_value3_num:누적획득량, 35:old_value3_num:과거량

def parse_datetime(_datetime):
    import datetime
    _parsed = datetime.datetime.strptime(str(_datetime),"%Y-%m-%d %H:%M:%S.%f")
    return _parsed

def append_amount(_dict, _key, _value):
    value = _dict.get(_key, 0)
    _dict[_key] = value + _value

for line in sys.stdin:
    if line.startswith("seq"): continue
    x = line.strip().split(",")
    _datetime = x[1]
    _parsed = parse_datetime(_datetime)

    log_id = int(x[2])
    key = _datetime[0:10]
    if log_id == LOG_EXP: exp = long(x[42])
    if log_id == LOG_MONEY: money = long(x[43])

    # 일 별 누적 경험치, 돈
    if exp > 0L: exp_amount[key] = exp
    if money > 0L: money_amount[key] = money
    append_amount(logs_amount, key, 1)

    # 일 별 누적 플레이타임
    if log_id == LOG_START:
        start_time = _datetime
    elif log_id == LOG_END and start_time != None:
        _start = parse_datetime(start_time)
        _end = parse_datetime(_datetime)
        _timedelta = _end - _start
        _seconds = _timedelta.seconds
        append_amount(play_millis, key, _seconds)
        start_time = None

print("Seconds", sorted(play_millis.items()))
print("Exp    ", sorted(exp_amount.items()))
print("Money  ", sorted(money_amount.items()))
print("Log    ", sorted(logs_amount.items()))

