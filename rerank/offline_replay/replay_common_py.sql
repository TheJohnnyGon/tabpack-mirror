$parseFlagsSets = Python::parse_flags_sets(
    Callable<(String?) -> List<List<Tuple<String, String>>>>,
    FileContent('helpers.py')
);

$pyUrlencode = Python::encode(Callable<(String?) -> String?>, FileContent('helpers.py'));

$pyUrlparse = Python::urlparse_wrapper(
    Callable<
        (String?) -> Struct<
            'Query': String,
            'Host': String,
            'Scheme': String,
            'Path': String,
            'Cgi': List<Tuple<String, String>>
        >?
    >, FileContent('helpers.py')
);

$makeQueryString = ($cgi) -> {
    RETURN String::JoinFromList(ListMap($cgi, ($tup) -> ($pyUrlencode($tup.0 ?? '') || '=' || $pyUrlencode($tup.1 ?? ''))), '&');
};

EXPORT
    $parseFlagsSets,
    $pyUrlencode,
    $pyUrlparse,
    $makeQueryString
;
Close
