import random
from datetime import datetime, timedelta, timezone

DEFAULT_DATE_FORMAT = "%Y-%m-%d"

wdays = {
    "en": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    "zh": [
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
        "星期日",
    ],
}


def get_date_now() -> list[int]:
    beijing_time = datetime.now(timezone.utc) + timedelta(hours=8)
    date = beijing_time.timetuple()
    date = [int(d) for d in [date.tm_year, date.tm_mon, date.tm_mday, date.tm_wday]]
    return date


def get_date_rand(before_days: int = 365, after_days: int = 365) -> list[int]:
    random_timedelta = timedelta(days=random.randint(-before_days, after_days))
    rand_time = datetime.now(timezone.utc) + random_timedelta
    date = rand_time.timetuple()
    date = [int(d) for d in [date.tm_year, date.tm_mon, date.tm_mday, date.tm_wday]]
    return date


def str2date(date_str, date_format: str = DEFAULT_DATE_FORMAT) -> list[int]:
    time = datetime.strptime(date_str, date_format)
    date = time.timetuple()
    date = [int(d) for d in [date.tm_year, date.tm_mon, date.tm_mday, date.tm_wday]]
    return date


def date2str(
    date,
    sep="-",
    with_week: bool = False,
    language: str = "en",
    date_format: str = DEFAULT_DATE_FORMAT,
) -> str:
    # YYYY-MM-DD
    if isinstance(date, str):
        date = str2date(date, date_format=date_format)

    date_str = sep.join([f"{date[0]:04d}", f"{date[1]:02d}", f"{date[2]:02d}"])
    if with_week:
        wday = wdays[language][date[3]]
        date_str = f"{date_str} {wday}"
    return date_str


if __name__ == "__main__":
    rand_date = get_date_now()
    print(date2str(rand_date, with_week=True))
    print(date2str(rand_date, with_week=True, language="zh"))
