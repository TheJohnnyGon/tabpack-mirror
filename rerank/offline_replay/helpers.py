from typing import List, Tuple, Any
import json
import re
import urllib.parse


def encode(s_):
    try:
        s_ = s_.decode("utf8")
    except:  # noqa
        return
    return urllib.parse.quote(s_).encode("utf8")


def urlparse_wrapper(str_):
    if not str_:
        return
    str_ = str_.decode("utf8", errors="replace")
    parsed = urllib.parse.urlparse(str_)
    cgi = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    return {
        "Query": parsed.query,
        "Host": parsed.netloc,
        "Scheme": parsed.scheme,
        "Path": parsed.path,
        "Cgi": cgi,
    }


def wrap_tup(tup: Tuple[str, Any]) -> List[Tuple[bytes, bytes]]:
    assert isinstance(tup[0], str)
    assert isinstance(tup[1], (str, int, float, list))
    if isinstance(tup[1], list):
        assert all(isinstance(x, str) for x in tup[1])
        result = []
        for x in tup[1]:
            result.append((tup[0].encode("utf8"), x.encode("utf8")))
        return result
    return [(tup[0].encode("utf8"), str(tup[1]).encode("utf8"))]


def dict_to_list_of_tuples(dict_):
    result = []
    for tup in dict_.items():
        result.extend(wrap_tup(tup))
    return result


def parse_inner_element(flags_set):
    if isinstance(flags_set, list):
        result = []
        for inner_dict in flags_set:
            assert isinstance(inner_dict, dict)
            result.extend(dict_to_list_of_tuples(inner_dict))
        return result
    elif isinstance(flags_set, dict):
        return dict_to_list_of_tuples(flags_set)
    else:
        raise Exception(f"flags_set must be either dict or list (of dicts), actually: {type(flags_set)}")


def parse_flags_sets(flags_sets):
    result = []
    if not flags_sets:
        return result
    flags_sets = flags_sets.decode("utf8")
    parsed = json.loads(flags_sets)
    if isinstance(parsed, dict):
        parsed = [parsed]
    assert isinstance(parsed, list)
    for flags_set in parsed:
        result.append(parse_inner_element(flags_set))
    return result


re_big_number = re.compile("[0-9]{9,}")
re_number = re.compile("[0-9]")


def check_not_weird(query):
    if not query:
        return False
    query = query.decode("utf8", errors="replace")
    if any(re_big_number.search(x) for x in query.split()):
        return False
    without_numbers = re_number.sub("", query).strip()
    if not without_numbers:
        return False
    return True


def get_url_for_soy(url, token):
    """
    Sometimes we find in geocube not an addrs.yandex.ru url
    but a maps proxy url (/maps/api/search).
    We need to retrieve the original addrs.yandex.ru urls in order to be able
    to compare them. It is done via querying the proxy once again
    with parameter &dump=request, also metrics token is added
    to bypass csrf check.
    """
    url = url.decode("utf8", errors="replace")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    old_qs = urllib.parse.parse_qsl(parsed.query)
    qs = [x for x in old_qs if x[0] != "csrfToken"]
    qs.append(("metricstoken", token.decode("utf8")))
    new_url = parsed._replace(netloc="l7test.yandex.ru", query=urllib.parse.urlencode(qs)).geturl() + "&dump=request"
    return new_url