from datetime import datetime

DATETIME_FORMAT = "%d/%m/%Y %H:%M"


def from_str(str_val):
    try:
        return int(str_val)
    except:
        pass

    try:
        return float(str_val)
    except:
        pass

    try:
        return datetime.strptime(str_val, DATETIME_FORMAT)
    except:
        pass

    return str_val


def to_str(val):
    if type(val) == datetime:
        return val.strftime(DATETIME_FORMAT)
    return str(val)


def read_csv(path):
    with open(path, mode="r") as f:
        lines = f.readlines()
        if lines:
            headers = lines[0].replace("\n", "").split(",")
            rows = []

            for l in lines[1:]:
                tmp_cols = l.replace("\n", "").split(",")
                rows.append(list(map(from_str, tmp_cols)))

            return headers, rows
        else:
            return [], []


def write_csv(path, headers, rows):
    with open(path, mode="w") as f:
        f.write(",".join(headers))
        f.write("\n")
        for r in rows:
            f.write(",".join(list(map(to_str, r))))
            f.write("\n")