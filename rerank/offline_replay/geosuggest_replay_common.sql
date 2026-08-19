$THRESHOLD_BY_POS = {
    1: 1.0,
    2: 0.95,
    3: 0.9,
    4: 0.85,
    5: 0.8,
    6: 0.75,
    7: 0.7,
};

$getHash = ($str) -> {
    $val = Digest::CityHash($str);
    RETURN CAST($val AS Double) / 18446744073709551615.0
};

$simulateClick = ($result, $userId, $sessionId, $planLat, $planLon, $distanceThreshold, $noRandom?) -> {
    $noRandom = $noRandom ?? FALSE;
    $isClose = Geo::CalculatePointsDifference($result.lat, $result.lon, $planLat, $planLon) <= $distanceThreshold;
    $pseudornd = $getHash(
        $userId || "___" || $sessionId || "___" || $result.shown_name || "___" || CAST($result.lat AS String) || "___" || CAST($result.lon AS String)
    );
    RETURN CASE
    WHEN NOT ($isClose ?? FALSE) THEN FALSE
    WHEN $noRandom OR ($pseudornd <= $THRESHOLD_BY_POS[$result.pos + 1ul]) THEN TRUE
    ELSE FALSE
    END
};

$getRightAnswerPos = ($results, $userId, $sessionId, $planLat, $planLon, $distanceThreshold, $noRandom?) -> {
    $noRandom = $noRandom ?? FALSE;
    $clickedPositions = ListNotNull(ListMap($results, ($result)->(IF(
        $simulateClick($result, $userId, $sessionId, $planLat, $planLon, $distanceThreshold, $noRandom),
        $result.pos
    ))));
    RETURN ListMin($clickedPositions)
};

EXPORT $simulateClick, $getRightAnswerPos;
Close
