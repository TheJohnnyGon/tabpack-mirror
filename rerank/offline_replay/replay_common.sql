PRAGMA yson.DisableStrict;

$patchCoordinates = ($coordinates) -> {
    $sp = String::SplitToList($coordinates, ',');
    $sp = ListNotNull(ListMap($sp, ($x) -> (CAST(String::Strip($x) AS Double))));
    $sp = ListMap($sp, ($x) -> (Math::Round($x, -6)));
    RETURN (CAST($sp[0] AS String) ?? '') || ',' || (CAST($sp[1] AS String) ?? '');
};

$SINGLE_SEP = '___';
$MULTI_SEP = ';;;';

$_makeId = ($elem) -> {
    $firstPart = CAST(TryMember($elem, 'Id', NULL) AS String) ?? (CAST(TryMember($elem, 'Name', NULL) AS String) ?? 'unknown');
    $secondPart = TryMember($elem, 'Type', NULL) ?? 'null';
    $firstPart = IF(
        $secondPart == 'COORDINATE',
        $patchCoordinates($firstPart),
        String::Strip($firstPart)
    );
    RETURN $firstPart || $SINGLE_SEP || $secondPart;
};

$makeId = ($answer) -> {
    $ids = ListNotNull(ListMap($answer.DocId.Elem, $_makeId));
    RETURN String::JoinFromList($ids, $MULTI_SEP);
};

$multiSpacesReplace = Re2::Replace(' +');

$normalize = ($requestText) -> {
    $requestText = Unicode::Fold(CAST($requestText AS Utf8));
    $requestText = Unicode::RemoveAll(Unicode::Fold($requestText), '.,');
    $requestText = $multiSpacesReplace($requestText, ' ');
    $requestText = String::Strip(CAST($requestText AS String));
    RETURN $requestText;
};

$parseDate = ($x) -> (DateTime::MakeDate(DateTime::Parse('%Y-%m-%d')($x)));
$formatDate = DateTime::Format('%Y-%m-%d');

$shiftDate = ($date, $days) -> (
    $formatDate(
        $parseDate($date) + DateTime::IntervalFromDays(CAST($days AS Int16))
    )
);

$getFirstDayOfMonth = ($date) -> ($shiftDate($date, 1 - CAST(DateTime::GetDayOfMonth($parseDate($date)) AS Int64)));
$mskDateFromTs = ($ts) -> ($formatDate(AddTimezone(DateTime::FromSeconds(CAST($ts AS Uint32)), 'Europe/Moscow')));

$mskDateFromDtWithSub = ($dt, $sub) -> {
    $dt = CAST($dt AS Datetime) - DateTime::IntervalFromMinutes($sub);
    $tm = DateTime::Split($dt);
    $tm = DateTime::MakeTimestamp(DateTime::Update($tm, 'Europe/Moscow' AS Timezone));
    RETURN $formatDate($tm);
};

$mskDateFromDt = ($dt) -> {
    $dt = CAST($dt AS Datetime);
    $tm = DateTime::Split($dt);
    $tm = DateTime::MakeTimestamp(DateTime::Update($tm, 'Europe/Moscow' AS Timezone));
    RETURN $formatDate($tm);
};

$daysDiff = ($date1, $date2) -> (DateTime::ToDays($date2 - $date1) + 1);

$dateRange = ($f, $t) -> {
    $f = $parseDate($f);
    $t = $parseDate($t);
    $diff = DateTime::ToDays($t - $f) + 1;
    RETURN ListMap(
        ListFromRange(0, Unwrap($diff)),
        ($x) -> (
            $formatDate($f + DateTime::IntervalFromDays(CAST($x AS Int16)))
        )
    );
};

$remapLogId = ($logId) -> {
    $firstElement = CAST($logId.source_id AS String) ?? $logId.name ?? 'EMPTY';
    $secondElement = String::AsciiToUpper($logId.type);
    $secondElement = CASE
        WHEN $secondElement == 'ORG1' THEN 'ORG'
        WHEN $secondElement == 'FAKE_CHAIN' THEN 'FAKECHAIN'
        WHEN $secondElement == 'COORDS' THEN 'COORDINATE'
        WHEN $secondElement == 'QUERY' THEN 'TEXT'
        ELSE $secondElement
    END;
    $firstElement = IF(
        $secondElement == 'COORDINATE',
        String::ReplaceAll($firstElement, ' ', ''),
        $firstElement
    );
    RETURN $firstElement || '___' || $secondElement;
};

$fsToString = ($fs) -> {
    $fs = ListMap(
        ListSort($fs), ($x) -> {
            $param = $x.0 ?? '';
            $value = String::ReplaceAll($x.1, ',', '__') ?? '';
            RETURN $param || '=' || $value;
        }
    );
    RETURN String::JoinFromList($fs, '&');
};

$parseSoyResultYql = ($fetchedResult) -> {
    $parsed = Yson::ParseJson($fetchedResult);
    $logIds = ListMap(
        Yson::ConvertToList(Yson::YPath($parsed, '/results')),
        ($x) -> (Yson::YPath($x, '/log_id'))
    );
    $parsedLogIds = ListMap(
        $logIds, ($x) -> (
            <|
                type: Yson::LookupString($x, 'type'),
                source_id: Yson::YPathString($x, '/WHERE/source_id') ?? Yson::YPathString($x, '/what/id'),
                name: Yson::YPathString($x, '/WHERE/name') ?? Yson::YPathString($x, '/what/name'),
            |>
        )
    );
    RETURN ListMap($parsedLogIds, $remapLogId);
};

$parseBasket = ($basket) -> (
    ListMap(
        Yson::ConvertToList(Yson::ParseJson($basket)),
        ($x) -> (
            <|
                country: Yson::LookupString($x, 'country'),
                table_type: Yson::LookupString($x, 'table_type'),
                weight: Yson::LookupDouble($x, 'weight')
            |>
        )
    )
);

$getLastDayOfMonth = ($date) -> {
    $date = CAST($date AS Date);
    $firstDayOfMonth = $date - DateTime::IntervalFromDays(DateTime::GetDayOfMonth($date) - 1);
    $guaranteedNextMonth = $firstDayOfMonth + DateTime::IntervalFromDays(32);
    RETURN CAST($guaranteedNextMonth - DateTime::IntervalFromDays(DateTime::GetDayOfMonth($guaranteedNextMonth)) AS String);
};

$checkClicks = ($answer) -> {
    $clicks = Yson::ConvertToList(Yson::Lookup($answer, 'clicks'));
    $gduTypes = ListNotNull(ListMap($clicks, ($x) -> (Yson::LookupString($x, 'gdu_type'))));
    RETURN ListLength($gduTypes) > 0;
};

$getGduPositions = ($aac) -> {
    $aac = Yson::ConvertToList($aac);
    $clicked = ListNotNull(
        ListMap(
            ListEnumerate($aac), ($tup) -> (
                IF($checkClicks($tup.1), $tup.0)
            )
        )
    );
    RETURN $clicked;
};

$w = ($x) -> (IF($x == '', NULL, $x));

$getDocumentId = ($result) -> {
    $type = Dsv::Parse(String::Base64Decode(Yson::LookupString($result, 'log_id')), ';')['type'] ?? '-';
    $id = Yson::LookupString($result, 'id');
    $text = Yson::LookupString($result, 'text');
    $base = Yson::LookupString($result, 'base');
    RETURN $base || ':' || $type || ':' || ($w($id) ?? $w($text));
};

$serpUrlIsGood = ($serpUrl) -> (
    FIND($serpUrl, '/api/search') IS NOT NULL
    OR (
        FIND($serpUrl, '/yandsearch') IS NOT NULL
        AND FIND($serpUrl, 'addrs.yandex.ru') IS NOT NULL
    )
);

$urlHash = ($url) -> (Digest::CityHash(($url ?? '') || 'somesalt'));

$parseGeosuccubeResults = ($results) -> (
    ListMap(
        ListEnumerate($results), ($tup) -> {
            $pos = $tup.0;
            $res = $tup.1;
            RETURN <|
                clicked: IF($res.Click IS NOT NULL, 1, 0),
                dedup_id: $makeId($res),
                lat: $res.Latitude,
                lon: $res.Longitude,
                pos: $pos,
                is_personal: $res.IsPersonal ?? FALSE,
                shown_name: $res.ShownName,
            |>;
        }
    )
);

$distanceKm = ($lat1, $lon1, $lat2, $lon2) -> (
    Geo::CalculatePointsDifference($lat1, $lon1, $lat2, $lon2) / 1000.0
);

$stripBaseUrl = ($fs) -> {
    $baseUrls = ListNotNull(ListMap(ListFilter($fs, ($x) -> ($x.0 == 'base_url')), ($x) -> ($x.1)));
    $notBaseUrls = ListNotNull(ListFilter($fs, ($x) -> ($x.0 != 'base_url')));
    RETURN Ensure(
        AsTuple($baseUrls, $notBaseUrls),
        ListLength($baseUrls) <= 1,
        'more than one base_url'
    );
};

$wrapUrl = ($url) -> (IF($url LIKE 'http%', $url, 'http://' || $url));

EXPORT
    $SINGLE_SEP,
    $MULTI_SEP,
    $makeId,
    $normalize,
    $parseDate,
    $formatDate,
    $shiftDate,
    $getFirstDayOfMonth,
    $getLastDayOfMonth,
    $mskDateFromTs,
    $mskDateFromDt,
    $mskDateFromDtWithSub,
    $daysDiff,
    $dateRange,
    $parseBasket,
    $fsToString,
    $parseSoyResultYql,
    $getGduPositions,
    $getDocumentId,
    $serpUrlIsGood,
    $w,
    $urlHash,
    $parseGeosuccubeResults,
    $distanceKm,
    $stripBaseUrl,
    $wrapUrl
;