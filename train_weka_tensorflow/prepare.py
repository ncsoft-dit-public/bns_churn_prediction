#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

ARFF = True
DEBUG = False
BREAKNO = 65535

answers = {}

def load_answer(filepath):
    _file = open(filepath, "r")
    for line in _file:
        uid, churn, value = line.strip().split(",")
        answer = "churn"
        if churn == "0": answer = "not_churn"
        answers[uid] = answer
    _file.close()

def parse_datetime(_datetime):
    import datetime
    _parsed = datetime.datetime.strptime(str(_datetime),"%Y-%m-%d %H:%M:%S.%f")
    return _parsed

def append_amount(_dict, _key, _value):
    value = _dict.get(_key, 0)
    _dict[_key] = value + _value

def split_by_key_value(_dict):
    keys = []
    values = []
    for _key, _value in sorted(_dict.items()):
        key = long(_key.replace('-', ''))
        value = long(_value)
        keys.append(key)
        values.append(value)
    return (keys, values)

def get_gradient(_dict):
    if len(_dict) == 0: return 0L
    x, y = split_by_key_value(_dict)
    from scipy import stats
    import numpy as np
    gradient,intercept,r_value,p_value,std_err=stats.linregress(x,y)
    return gradient

def read_data(filepath):
    exp = 0L
    money = 0L
    start_time = None   # 시작세션 시간

    playtime_millis = {}    # 1003번이 발견되면 start_time 업데이트, 1004번이 발견되면 start_time 시간이 있으면 비교하여 diff_millis 값을 yyyymmdd 킷값에 amount 값을 늘린다
    exp_amount = {}     # 일자별 경험치 스냅샷 - 누적 획득량을 계속 업데이트하면 끗
    money_amount = {}   # 일자별 돈 스냅샷 - 누적획득량을 계속 업데이트하면 끗
    logs_amount = {}    # 일 별 로그의 빈도수

    LOG_START = 1003      # 2:  log_id
    LOG_END = 1004        # 2:  log_id
    LOG_EXP = 1016        # 37: use_value1_num:획득량, 41:new_value2_num:누적획득량
    LOG_MONEY = 1017      # 38: use_value2_num:획득량, 42:new_value3_num:누적획득량, 35:old_value3_num:과거량

    _file = open(filepath, "r")
    for line in _file:
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
            append_amount(playtime_millis, key, _seconds)
            start_time = None
    _file.close()
    if DEBUG:
        print("Playtime	", sorted(playtime_millis.items()))
        print("Exp    	", sorted(exp_amount.items()))
        print("Money  	", sorted(money_amount.items()))
        print("Log    	", sorted(logs_amount.items()))
    playtime_trends = get_gradient(playtime_millis)
    exp_trends = get_gradient(exp_amount)
    money_trends = get_gradient(money_amount)
    logs_trends = get_gradient(logs_amount)
    return (playtime_trends, exp_trends, money_trends, logs_trends)

arff = """
@RELATION bns_churn_detection

@ATTRIBUTE playtime_trends  REAL
@ATTRIBUTE exp_trends       REAL
@ATTRIBUTE money_trends     REAL
@ATTRIBUTE log_trends       REAL
@ATTRIBUTE churn            {churn,not_churn,undefined}

@DATA
%s

%%
%%
%%
"""

def traverse_dir(path):
    strings = []
    from os import listdir
    lineno = 1
    for filepath in listdir(path):
        filename = filepath.split(".")[0]
        file_ext = filepath.split(".")[1]
        if file_ext != "csv": continue
        out = read_data(path + "/" + filepath)
        answer = answers.get(filename, "undefined")
        strings.append(out + (answer,))
        lineno += 1
        if lineno > BREAKNO: break
        if (lineno % 10) == 0: sys.stderr.write(".")
    return strings

def main():
    try:
        load_answer("train_labeld.csv")
        strings = traverse_dir(sys.argv[1])
        x = ""
        for string in strings:
            x += "%f,%f,%f,%f,%s\n" % string
        if ARFF:
            print(arff % x)
        else:
            print(x)
    except:
        import traceback
        traceback.print_exc(sys.stderr)

def exit():
    sys.exit(0)

def print_usage():
    print("python prepare.py data weka > ~/workspace/weka/bns_churn_detection.arff")
    print("python prepare.py data tf > ~/workspace/weka/bns_churn_detection.csv")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        exit()
    if sys.argv[2] == "tf":
        ARFF = False
    main()

