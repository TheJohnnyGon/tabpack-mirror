# План: сборка серпсет-таблицы (формат близкий к 65710690.json)

## 0. Уточнения от пользователя (финальные)

1. **`id` в корзинке** = `CAST(ROW_NUMBER() OVER() AS String)`; `cgi = requests[0].cgi`, `meta_info = requests`.
2. **`ranged_docs` НЕ отсортирован.** У каждой строки есть `DocId` (JSON документа) и `RawFormulaVal`. Сортируем по `RawFormulaVal DESC` внутри `GroupId` сами.
3. **md5 / organizationId — НЕ используем** (их нет в корзинке).
4. **`texts.labels`** — оставляем пустой массив `[]`.
5. **Выход = таблица с полями** (struct-колонки). В JSON пользователь перегонит сам.
6. **`GroupId` — ЭТО ТЕКСТ ЗАПРОСА.** Используем `GroupId` напрямую как `query.text`. Скобки не трогаем.

## 1. Что есть в источниках

| Файл | Роль | Поля |
|------|------|------|
| [`ranged_docs.json`](serp/ranged_docs.json) | Итог модели (НЕотсортирован) | `DocId(JSON)`, `GroupId(=query text)`, `Label`, `RawFormulaVal` |
| [`fetch_input.json`](serp/fetch_input.json) | Раскрутка (мост) | `CustomRequestId(=basket.id)`, `parsed_serps=[query, docs]` |
| [`basket_2.json`](serp/basket_2.json) | Корзинка после SQL | `id`, `cgi(список пар)`, `meta_info(=requests)` |

`DocId` (JSON) содержит: `lat, lon, subtitle, title, type, uri, where_name`.

## 2. Ключи джойна (подтверждено)

```
ranged_docs.GroupId  ==  fetch.parsed_serps.query        (текст запроса)
fetch.CustomRequestId  ==  basket_2.id                   (ROW_NUMBER-id)
```

`GroupId` используем и как `query.text`, и как ключ группировки компонентов.

Мост `fetch` нужен ТОЛЬКО чтобы получить `basket.id` по `GroupId` и дальше достать `cgi`
(из него `ull`→mapInfo, `client_reqid`→uid). Если метаданные запроса не нужны —
мост и корзинку можно не подключать (см. Вариант A).

## 3. Итоговая схема серпсет-таблицы (одна строка = один запрос)

| Колонка | Источник | Примечание |
|---------|----------|------------|
| `query.text` | `ranged_docs.GroupId` | напрямую |
| `query.mapInfo.x` (lon) | `basket.cgi.ull` | split "lon,lat"[0] |
| `query.mapInfo.y` (lat) | `basket.cgi.ull` | split "lon,lat"[1] |
| `query.mapInfo.spnX/spnY/zoom` | — | дефолты 0.0/0.0/0 (нет в cgi) |
| `query.uid` | `basket.cgi.client_reqid` | |
| `query.device` | — | дефолт 1 (нет в cgi) |
| `query.regionId` | — | нет в корзинке после SQL → опустить/null |
| `query.country` | — | нет в корзинке после SQL → опустить/null |
| `serpInfo` | — | заглушка {0,0,-1} |
| `texts.labels` | — | пустой массив `[]` |
| `type` | — | "SERP" |
| `components[]` | `ranged_docs` | сорт по RawFormulaVal DESC |

### Компонент (`components[]`)
| Поле | Источник (из DocId) |
|------|---------------------|
| `componentInfo` | {type:1, alignment:3} (константы) |
| `componentUrl.pageUrl` | `uri` |
| `coordinates.coordinates.longitude` | `lon` |
| `coordinates.coordinates.latitude` | `lat` |
| `text.title` | `title` |
| `text.snippet` | `subtitle` |
| `text.rubrics` | "" |
| `type` | "COMPONENT" |

## 4. Два варианта реализации

### Вариант A (минимальный) — только ranged_docs
Один вход `$input1 = ranged_docs`. Собираем `query.text = GroupId` + `components`
(сортировка по RawFormulaVal DESC). Метаданные запроса (mapInfo/uid) пустые/дефолт.
Годится, если инфраструктуре достаточно текста запроса + компонентов.

### Вариант B (полный) — с обогащением из корзинки
Три входа: `$input1 = ranged_docs`, `$input2 = fetch`, `$input3 = basket_2`.
Джойн `ranged_docs.GroupId → fetch.parsed_serps.query → fetch.CustomRequestId
→ basket.id → cgi`. Из `cgi` берём `ull`(mapInfo) и `client_reqid`(uid).

## 5. Черновик SQL — Вариант A (минимальный)

```sql
PRAGMA yt.InferSchema = '1';
DECLARE $input1 AS String;   -- ranged_docs
DECLARE $output1 AS String;

$parsed = (
    SELECT
        GroupId AS query_text,
        CAST(RawFormulaVal AS Double) AS score,
        Yson::ParseJson(CAST(DocId AS String)) AS doc
    FROM $input1
);

$fields = (
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

INSERT INTO $output1 WITH TRUNCATE
SELECT
    AsStruct(query_text AS text)          AS `query`,
    AsStruct(0 AS responseTime, 0 AS responseSize, -1 AS notAnsweredHostCount) AS serpInfo,
    AsList()                              AS `texts.labels`,   -- пустой массив
    "SERP"                                AS type,
    ListSort(
        AGGREGATE_LIST(
            AsStruct(
                AsStruct(1 AS type, 3 AS alignment)          AS componentInfo,
                AsStruct(uri AS pageUrl)                     AS componentUrl,
                AsStruct(lon AS longitude, lat AS latitude)  AS `coordinates.coordinates`,
                title                                        AS `text.title`,
                subtitle                                     AS `text.snippet`,
                ""                                           AS `text.rubrics`,
                "COMPONENT"                                  AS type,
                score                                        AS _score
            )
        ),
        ($x) -> { RETURN -$x._score; }   -- RawFormulaVal DESC
    )                                     AS components
FROM $fields
GROUP BY query_text;
```

## 6. Черновик SQL — Вариант B (с обогащением из корзинки)

```sql
PRAGMA yt.InferSchema = '1';
DECLARE $input1 AS String;   -- ranged_docs
DECLARE $input2 AS String;   -- fetch (parsed_serps + CustomRequestId)
DECLARE $input3 AS String;   -- basket_2 (id, cgi, meta_info)
DECLARE $output1 AS String;

-- 6.1 компоненты по запросу (как в Варианте A)
$parsed = (
    SELECT GroupId AS query_text,
           CAST(RawFormulaVal AS Double) AS score,
           Yson::ParseJson(CAST(DocId AS String)) AS doc
    FROM $input1
);
$fields = (
    SELECT query_text, score,
           Yson::ConvertToDouble(doc.lat)      AS lat,
           Yson::ConvertToDouble(doc.lon)      AS lon,
           Yson::ConvertToString(doc.title)    AS title,
           Yson::ConvertToString(doc.subtitle) AS subtitle,
           Yson::ConvertToString(doc.uri)      AS uri
    FROM $parsed
);
$components = (
    SELECT query_text,
        ListSort(
            AGGREGATE_LIST(AsStruct(
                AsStruct(1 AS type, 3 AS alignment)         AS componentInfo,
                AsStruct(uri AS pageUrl)                    AS componentUrl,
                AsStruct(lon AS longitude, lat AS latitude) AS `coordinates.coordinates`,
                title AS `text.title`, subtitle AS `text.snippet`,
                "" AS `text.rubrics`, "COMPONENT" AS type, score AS _score
            )),
            ($x) -> { RETURN -$x._score; }
        ) AS components
    FROM $fields GROUP BY query_text
);

-- 6.2 мост: query_text -> basket.id (через fetch)
$bridge = (
    SELECT
        Yson::ConvertToString(parsed_serps[0]) AS query_text,
        CAST(CustomRequestId AS String)        AS req_id
    FROM $input2
);

-- 6.3 данные запроса из корзинки (cgi -> ull, uid)
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

INSERT INTO $output1 WITH TRUNCATE
SELECT
    AsStruct(
        c.query_text AS text,
        AsStruct(
            CAST(ListHead(String::SplitToList(m.ull, ",")) AS Double) AS x,
            CAST(ListLast(String::SplitToList(m.ull, ",")) AS Double) AS y,
            0.0 AS spnX, 0.0 AS spnY, 0 AS zoom
        ) AS mapInfo,
        m.uid AS uid,
        1     AS device
    )                                 AS `query`,
    AsStruct(0 AS responseTime, 0 AS responseSize, -1 AS notAnsweredHostCount) AS serpInfo,
    AsList()                          AS `texts.labels`,
    "SERP"                            AS type,
    c.components                      AS components
FROM $components AS c
LEFT JOIN $bridge   AS b ON c.query_text == b.query_text
LEFT JOIN $query_meta AS m ON b.req_id == m.req_id;
```

## 7. Замечания / риски

1. **Дубли в ranged_docs.** Есть идентичные строки (один DocId несколько раз). Не
   дедуплицирую — сортировка по score сохранит все. Если нужно уникально — добавить
   `DISTINCT` по DocId внутри группы.
2. **regionId/country.** После SQL корзинки (#1) их нет ни в `cgi`, ни в `meta_info`.
   Поэтому в схему не включены. Если нужны — тянуть из исходной таблицы корзинки
   (там `request_region`, `country` на верхнем уровне) отдельным входом.
3. **spn/zoom.** В `cgi` нет `spn` → mapInfo.spnX/spnY = 0.0. Fallback возможен из
   fetch (`FetchedResult.log_id.user_params.spn`), но это парсинг вложенного JSON.
4. **Имена полей с точкой** (`text.title`, `coordinates.coordinates`) — так в целевом
   формате; в YQL берём в обратные кавычки. Если сериализатор ждёт вложенность —
   поменяем на вложенные структуры.
5. **ListSort по score.** Отдельное поле `_score` внутри компонента для сортировки;
   при финальной сериализации его можно отбросить.

## 8. Что нужно решить перед материализацией .sql

- Вариант A (только ranged_docs) или B (с обогащением из корзинки)?
- Нужны ли regionId/country (тянуть из исходной корзинки доп. входом)?
- Дедупликация документов внутри запроса — да/нет?
