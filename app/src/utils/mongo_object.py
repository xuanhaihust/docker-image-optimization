# some util functions for process object in mongodb, such as ObjectId...

def str_ojb_id(d: dict) -> dict:
    # convert all bson ObjectId in a dict to str
    try:
        for k, v in d.items():
            if k == '_id':
                d[k] = str(v)
            else:
                if isinstance(v, dict):
                    str_ojb_id(v)
                elif isinstance(v, list):
                    for _l in v:
                        if isinstance(_l, dict):
                            str_ojb_id(_l)
    except Exception as e:
        raise e
    return d