PRAGMA yt.InferSchema = '1';
DECLARE $input1 AS String;   -- ranged_docs: DocId(JSON), GroupId, Label, RawFormulaVal
DECLARE $input2 AS String;   -- fetch: parsed_serps=[query, docs], CustomRequestId
DECLARE $input3 AS String;   -- basket_2: id, cgi(список пар), meta_info
DECLARE $output1 AS String;  -- итоговая таблица серпсетов

-- =====================================================================
-- ШАГ 1. Распарсить DocId (JSON-строку) в поля документа
-- =====================================================================

$parsed = (
    SELECT
        GroupId AS query_text,
        CAST(RawFormulaVal AS Double) AS score,
        Yson::ParseJson(CAST(DocId AS String)) AS doc
    FROM $input1
);

$doc_fields = (
    SELECT
        query_text,
        score,
        Yson::ConvertToDouble(doc.lat)      AS lat,
        Yson::ConvertToDouble(doc.lon)      AS lon,
        Yson::ConvertToString(doc.title)    AS title,
        Yson::ConvertToString(doc.subtitle) AS subtitle,
        Yson::ConvertToString(doc.type)     AS ctype,
        Yson::ConvertToString(doc.uri)      AS uri
    FROM $parsed
);

-- =====================================================================
-- ШАГ 2. Собрать компоненты по запросу, отсортированные по score DESC
-- =====================================================================

$components = (
    SELECT
        query_text,
        ListSort(
            AGGREGATE_LIST(
                AsStruct(
                    AsStruct(1 AS type, 3 AS alignment)              AS componentInfo,
                    AsStruct(uri AS pageUrl)                          AS componentUrl,
                    AsStruct(lon AS longitude, lat AS latitude)       AS coordinates,
                    title                                             AS text_title,
                    subtitle                                          AS text_snippet,
                    ""                                                AS text_rubrics,
                    "COMPONENT"                                       AS type,
                    score                                             AS _score
                )
            ),
            ($x) -> { RETURN -$x._score; }  -- сортировка по score DESC
        ) AS components
    FROM $doc_fields
    GROUP BY query_text
);

-- =====================================================================
-- ШАГ 3. Мост: query_text → basket.id (через fetch)
-- =====================================================================

$bridge = (
    SELECT
        Yson::ConvertToString(parsed_serps[0]) AS query_text,
        CAST(CustomRequestId AS String)        AS req_id
    FROM $input2
);

-- =====================================================================
-- ШАГ 4. Данные запроса из корзинки (cgi → ull, uid)
-- =====================================================================

$cgi_val = ($cgi, $name) -> {
    RETURN ListHead(ListNotNull(ListMap($cgi, ($p) -> {
        RETURN IF(Yson::ConvertToString($p[0]) == $name,
                  Yson::ConvertToString($p[1]), NULL);
    })));
};

$query_meta = (
    SELECT
        CAST(id AS String) AS req_id,
        $cgi_val(cgi, "client_reqid") AS uid,
        $cgi_val(cgi, "ull")          AS ull
    FROM $input3
);

-- =====================================================================
-- ШАГ 5. Финальная сборка: JOIN компонентов с метаданными запроса
-- =====================================================================

INSERT INTO $output1 WITH TRUNCATE
SELECT
    -- query: text + mapInfo (из ull) + uid (из client_reqid)
    AsStruct(
        c.query_text AS text,
        AsStruct(
            CAST(ListHead(String::SplitToList(m.ull, ",")) AS Double) AS x,
            CAST(ListLast(String::SplitToList(m.ull, ",")) AS Double) AS y,
            0.0 AS spnX,
            0.0 AS spnY,
            0   AS zoom
        ) AS mapInfo,
        m.uid AS uid,
        1     AS device
    ) AS `query`,
    
    -- serpInfo: заглушка
    AsStruct(
        0  AS responseTime,
        0  AS responseSize,
        -1 AS notAnsweredHostCount
    ) AS serpInfo,
    
    -- texts: структура с полем labels (пустой массив)
    AsStruct(AsList() AS labels) AS texts,
    
    -- type: SERP
    "SERP" AS type,
    
    -- components: массив документов, отсортированный по RawFormulaVal DESC
    c.components AS components
FROM $components AS c
LEFT JOIN $bridge     AS b ON c.query_text == b.query_text
LEFT JOIN $query_meta AS m ON b.req_id == m.req_id;
